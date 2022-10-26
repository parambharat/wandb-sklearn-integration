import functools
import inspect
import logging
import weakref
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Union

import numpy as np
from packaging.version import Version

import wandb
from integration.sklearn.autologging_utils import (
    _get_new_training_session_class,
    disable_autologging,
)
from integration.sklearn.utils import (
    log_model,
    _SklearnCustomModelPicklingError,
    _inspect_original_var_name,
    safe_patch,
)

_logger = logging.getLogger(__name__)

SERIALIZATION_FORMAT_PICKLE = "pickle"
INPUT_EXAMPLE_SAMPLE_ROWS = 5

_SklearnTrainingSession = _get_new_training_session_class()

# The `_apis_autologging_disabled` contains APIs which is incompatible with autologging,
# when user call these APIs, autolog is temporarily disabled.
_apis_autologging_disabled = [
    "cross_validate",
    "cross_val_predict",
    "cross_val_score",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
]


class _AutologgingMetricsManager:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
       Note: We need this map instead of setting the run_id into the "prediction result object"
       because the object maybe a numpy array which does not support additional attribute
       assignment.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `model.fit`, `eval_and_log_metrics`, `model.score`,
       in order to avoid nested/duplicated autologging metric, when run into these scopes,
       we need temporarily disable the metric autologging.
    (3) _eval_dataset_info_map, it is a double level map:
       `_eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
    (4) _metric_api_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` wil get a list of tuples, each tuple is:
         (logged_metric_key, metric_call_command_string)
        each call command string is like `metric_fn(arg1, arg2, ...)`
        This data structure is used for:
         * storing the call arguments dict for each metric call, we need log them into metric_info
           artifact file.

    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only hold IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """

    def __init__(self):
        self._pred_result_id_to_dataset_name_and_run_id = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._metric_api_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safeguard checking,
        See following note.

        Note: It includes checking `_SklearnTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. GridSearchCV) case:
          running GridSearchCV.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return (
                not _SklearnTrainingSession.is_active()
                and self._log_post_training_metrics_enabled
        )

    def disable_log_post_training_metrics(self):
        class LogPostTrainingMetricsDisabledScope:
            def __enter__(inner_self):  # pylint: disable=no-self-argument
                # pylint: disable=attribute-defined-outside-init
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            # pylint: disable=no-self-argument
            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status

        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_id_for_model(model):
        return getattr(model, "_wandb_run_id", None)

    @staticmethod
    def is_metric_value_loggable(metric_value):
        """
        check whether the specified `metric_value` is a numeric value which can be logged
        as a wandb metric.
        """
        return isinstance(metric_value, (int, float, np.number)) and not isinstance(
            metric_value, bool
        )

    @staticmethod
    def register_model(model, run_id):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._wandb_run_id = run_id

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            # Use '-' as the separator between name and index,
            # The '-' is not valid character in python var name
            # so, it can prevent name conflicts after appending index.
            return f"{name}-{index + 1}"

    def register_prediction_input_dataset(self, model, eval_dataset):
        """
        Register prediction input dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id.
         3. register eval dataset with id.
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
        eval_dataset_name = _inspect_original_var_name(
            eval_dataset, fallback_name="unknown_dataset"
        )
        eval_dataset_id = id(eval_dataset)

        run_id = self.get_run_id_for_model(model)
        registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]

        for i, id_i in enumerate(registered_dataset_list):
            if eval_dataset_id == id_i:
                index = i
                break
        else:
            index = len(registered_dataset_list)

        if index == len(registered_dataset_list):
            # register new eval dataset
            registered_dataset_list.append(eval_dataset_id)

        return self.gen_name_with_index(eval_dataset_name, index)

    def register_prediction_result(self, run_id, eval_dataset_name, predict_result):
        """
        Register the relationship
         id(prediction_result) --> (eval_dataset_name, run_id)
        into map `_pred_result_id_to_dataset_name_and_run_id`
        """
        value = (eval_dataset_name, run_id)
        prediction_result_id = id(predict_result)
        self._pred_result_id_to_dataset_name_and_run_id[prediction_result_id] = value

        def clean_id(id_):
            _AUTOLOGGING_METRICS_MANAGER._pred_result_id_to_dataset_name_and_run_id.pop(
                id_, None
            )

        # When the `predict_result` object being GCed, its ID may be reused, so register a finalizer
        # to clear the ID from the dict for preventing wrong ID mapping.
        weakref.finalize(predict_result, clean_id, prediction_result_id)

    @staticmethod
    def gen_metric_call_command(self_obj, metric_fn, *call_pos_args, **call_kwargs):
        """
        Generate metric function call command string like `metric_fn(arg1, arg2, ...)`
        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.

        param self_obj: If the metric_fn is a method of an instance (e.g. `model.score`),
           the `self_obj` represent the instance.
        param metric_fn: metric function.
        param call_pos_args: the positional arguments of the metric function call. If `metric_fn`
          is instance method, then the `call_pos_args` should exclude the first `self` argument.
        param call_kwargs: the keyword arguments of the metric function call.
        """

        arg_list = []

        def arg_to_str(arg):
            if arg is None or np.isscalar(arg):
                if isinstance(arg, str) and len(arg) > 32:
                    # truncate too long string
                    return repr(arg[:32] + "...")
                return repr(arg)
            else:
                # dataset arguments or other non-scalar type argument
                return _inspect_original_var_name(
                    arg, fallback_name=f"<{arg.__class__.__name__}>"
                )

        param_sig = inspect.signature(metric_fn).parameters
        arg_names = list(param_sig.keys())

        if self_obj is not None:
            # If metric_fn is a method of an instance, e.g. `model.score`,
            # then the first argument is `self` which we need exclude it.
            arg_names.pop(0)

        if self_obj is not None:
            call_fn_name = f"{self_obj.__class__.__name__}.{metric_fn.__name__}"
        else:
            call_fn_name = metric_fn.__name__

        # Attach param signature key for positional param values
        for arg_name, arg in zip(arg_names, call_pos_args):
            arg_list.append(f"{arg_name}={arg_to_str(arg)}")

        for arg_name, arg in call_kwargs.items():
            arg_list.append(f"{arg_name}={arg_to_str(arg)}")

        arg_list_str = ", ".join(arg_list)

        return f"{call_fn_name}({arg_list_str})"

    def register_metric_api_call(self, run_id, metric_name, dataset_name, call_command):
        """
        This method will do:
        (1) Generate and return metric key, format is:
          {metric_name}[-{call_index}]_{eval_dataset_name}
          metric_name is generated by metric function name, if multiple calls on the same
          metric API happen, the following calls will be assigned with an increasing "call index".
        (2) Register the metric key with the "call command" information into
          `_AUTOLOGGING_METRICS_MANAGER`. See doc of `gen_metric_call_command` method for
          details of "call command".
        """

        call_cmd_list = self._metric_api_call_info[run_id][metric_name]

        index = len(call_cmd_list)
        metric_name_with_index = self.gen_name_with_index(metric_name, index)
        metric_key = f"{metric_name_with_index}_{dataset_name}"

        call_cmd_list.append((metric_key, call_command))

        # Set the flag to true, represent the metric info in this run need update.
        # Later when `log_eval_metric` called, it will generate a new metric_info artifact
        # and overwrite the old artifact.
        self._metric_info_artifact_need_update[run_id] = True
        return metric_key

    def get_run_id_and_dataset_name_for_metric_api_call(
            self, call_pos_args, call_kwargs
    ):
        """
        Given a metric api call (include the called metric function, and call arguments)
        Register the call information (arguments dict) into the `metric_api_call_arg_dict_list_map`
        and return a tuple of (run_id, eval_dataset_name)
        """
        call_arg_list = list(call_pos_args) + list(call_kwargs.values())

        dataset_id_list = self._pred_result_id_to_dataset_name_and_run_id.keys()

        # Note: some metric API the arguments is not like `y_true`, `y_pred`
        #  e.g.
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
        for arg in call_arg_list:
            if arg is not None and not np.isscalar(arg) and id(arg) in dataset_id_list:
                dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[
                    id(arg)
                ]
                break
        else:
            return None, None

        return run_id, dataset_name

    def log_post_training_metric(self, run_id, key, value):
        """
        Log the metric into the specified wandb run.
        and it will also update the metric_info artifact if needed.
        """
        # Note: if the case log the same metric key multiple times,
        #  newer value will overwrite old value
        _run = wandb.run if wandb.run is not None else wandb.init(id=run_id, )
        _run.log({key: value})
        if self._metric_info_artifact_need_update[run_id]:
            call_commands_list = []
            for v in self._metric_api_call_info[run_id].values():
                call_commands_list.extend(v)

            call_commands_list.sort(key=lambda x: x[0])
            dict_to_log = OrderedDict(call_commands_list)
            _run.log(dict_to_log, )
            self._metric_info_artifact_need_update[run_id] = False


# The global `_AutologgingMetricsManager` instance which holds information used in
# post-training metric autologging. See doc of class `_AutologgingMetricsManager` for details.
_AUTOLOGGING_METRICS_MANAGER = _AutologgingMetricsManager()

_metric_api_excluding_list = ["check_scoring", "get_scorer", "make_scorer"]


def autolog(
        flavour_name: str = "sklearn",
        log_input_examples: bool = False,
        log_models: bool = True,
        disable: bool = False,
        exclusive: bool = False,
        disable_for_unsupported_versions: bool = False,
        silent: bool = False,
        max_tuning_runs: int = 5,
        log_post_training_metrics: bool = True,
        serialization_format: str = SERIALIZATION_FORMAT_PICKLE,
        pos_label: Union[str, int] = None,
):
    """
    Enables (or disables) and configures autologging for scikit-learn estimators.

    **When is autologging performed?**
      Autologging is performed when you call:

      - ``estimator.fit()``
      - ``estimator.fit_predict()``
      - ``estimator.fit_transform()``

    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.get_params(deep=True)``. Note that ``get_params``
          is called with ``deep=True``. This means when you fit a meta estimator that chains
          a series of estimators, the parameters of these child estimators are also logged.

      **Training metrics**
        - A training score obtained by ``estimator.score``. Note that the training score is
          computed using parameters given to ``fit()``.
        - Common metrics for classifier:

          - `precision score`_

          .. _precision score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

          - `recall score`_

          .. _recall score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

          - `f1 score`_

          .. _f1 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

          - `accuracy score`_

          .. _accuracy score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

          If the classifier has method ``predict_proba``, we additionally log:

          - `log loss`_

          .. _log loss:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

          - `roc auc score`_

          .. _roc auc score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

        - Common metrics for regressor:

          - `mean squared error`_

          . _mean squared error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

          - root mean squared error

          - `mean absolute error`_

          . _mean absolute error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

          - `r2 score`_

          . _r2 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

      .. _post training metrics:

      **Post training metrics**
        When users call metric APIs after model training, wandb tries to capture the metric API
        results and log them as wandb metrics to the Run associated with the model. The following
        types of scikit-learn metric APIs are supported:

        - model.score
        - metric APIs defined in the `sklearn.metrics` module

        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"

        - If the metric function is from `sklearn.metrics`, the wandb "metric_name" is the
          metric function name. If the metric function is `model.score`, then "metric_name" is
          "{model_class_name}_score".
        - If multiple calls are made to the same scikit-learn metric API, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - wandb uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the first argument of the associated `model.predict` or `model.score` call.
          Note: wandb captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.

        **Limitations**
           - wandb can only map the original prediction result object returned by a model
             prediction API (including predict / predict_proba / predict_log_proba / transform,
             but excluding fit_predict / fit_transform.) to a wandb run.
             wandb cannot find run information
             for other objects derived from a given prediction result (e.g. by copying or selecting
             a subset of the prediction result). scikit-learn metric APIs invoked on derived objects
             do not log metrics to wandb.
           - Autologging must be enabled before scikit-learn metric APIs are imported from
             `sklearn.metrics`. Metric APIs imported before autologging is enabled do not log
             metrics to wandb runs.
           - If user define a scorer which is not based on metric APIs in `sklearn.metrics`,
             then post training metric autologging for the scorer is invalid.

        **Tags**
          - An estimator class name (e.g. "LinearRegression").
          - A fully qualified estimator class name
            (e.g. "sklearn.linear_model._base.LinearRegression").

        **Artifacts**
          - An wandb Model with the :py:mod:`wandb.sklearn` flavor containing a fitted estimator
            (logged by :py:func:`wandb.sklearn.log_model()`). The Model also contains the
            :py:mod:`wandb.pyfunc` flavor when the scikit-learn estimator defines `predict()`.
          - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
            JSON object whose keys are wandb post training metric names
            (see "Post training metrics" section for the key format) and whose values are the
            corresponding metric call commands that produced the metrics, e.g.
            ``accuracy_score(y_true=test_iris_y, y_pred=pred_iris_y, normalize=False)``.

    **How does autologging work for meta estimators?**
      When a meta estimator (e.g. `Pipeline`_, `GridSearchCV`_) calls ``fit()``, it internally calls
      ``fit()`` on its child estimators. Autologging does NOT perform logging on these constituent
      ``fit()`` calls.

      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`GridSearchCV`_ and `RandomizedSearchCV`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model (if available).

    **Supported estimators**
      - All estimators obtained by `sklearn.utils.all_estimators`_ (including meta estimators).
      - `Pipeline`_
      - Parameter search estimators (`GridSearchCV`_ and `RandomizedSearchCV`_)

    .. _sklearn.utils.all_estimators:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html

    .. _Pipeline:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    .. _GridSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    .. _RandomizedSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html




    param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with scikit-learn model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are wandb model attributes
                               and are only collected if ``log_models`` is also ``True``.
    param log_models: If ``True``, trained models are logged as wandb artifact.
                       If ``False``, trained models are not logged.

    param disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
                    enables the scikit-learn autologging integration.
    param exclusive: If ``True``, autologged content is not logged to user-created wandb runs.
                      If ``False``, autologged content is logged to the active wandb run,
                      which may be user-created.
    param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      scikit-learn that have not been tested against this version of the wandb
                      client or are incompatible.
    param silent: If ``True``, suppress all event logs and warnings from wandb during scikit-learn
                   autologging. If ``False``, show all events and warnings during scikit-learn
                   autologging.
    param max_tuning_runs: The maximum number of child wandb runs created for hyperparameter
                            search estimators. To create child runs for the best `k` results from
                            the search, set `max_tuning_runs` to `k`. The default value is to track
                            the best 5 search parameter sets. If `max_tuning_runs=None`, then
                            a child run is created for each search parameter set. Note: The best k
                            results is based on ordering in `rank_test_score`. In the case of
                            multi-metric evaluation with a custom scorer, the first scorer’s
                            `rank_test_score_<scorer_name>` will be used to select the best k
                            results. To change metric used for selecting best k results, change
                            ordering of dict passed as `scoring` parameter for estimator.
    param log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
                                      ``True``. See the `post training metrics`_ section for more
                                      details.
    param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``wandb.integration.sklearn.SERIALIZATION_FORMAT_PICKLE``.
    param pos_label: If given, used as the positive label to compute binary classification
                      training metrics such as precision, recall, f1, etc. This parameter should
                      only be set for binary classification model. If used for multi-label model,
                      the training metrics calculation will fail and the training metrics won't
                      be logged. If used for regression model, the parameter will be ignored.
    """
    _autolog(
        flavour_name=flavour_name,
        log_input_examples=log_input_examples,
        log_models=log_models,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        max_tuning_runs=max_tuning_runs,
        log_post_training_metrics=log_post_training_metrics,
        serialization_format=serialization_format,
        pos_label=pos_label,
    )


def _autolog(
        flavour_name: str = "sklearn",
        log_input_examples: bool = False,
        log_models: bool = True,
        disable: bool = False,
        exclusive: bool = False,
        disable_for_unsupported_versions: bool = False,
        silent: bool = False,
        max_tuning_runs: int = 5,
        log_post_training_metrics: int = True,
        serialization_format: str = SERIALIZATION_FORMAT_PICKLE,
        pos_label: Union[int, str] = None,
):
    import pandas as pd
    import sklearn.metrics
    import sklearn.model_selection

    from integration.sklearn.utils import (
        _TRAINING_PREFIX,
        _get_x_y_and_sample_weight,
        _log_estimator_content,
        _all_estimators,
        _get_estimator_info_tags,
        _get_meta_estimators_for_autologging,
        _is_parameter_search_estimator,
        _log_parameter_search_results_as_artifact,
        _is_metric_value_loggable,
        update_wrapper_extended,
        _gen_estimators_to_patch,
        _patch_estimator_method_if_available,
        _get_instance_method_first_arg_value,
        _get_metric_name_list,
    )

    if max_tuning_runs is not None and max_tuning_runs < 0:
        raise wandb.Error(
            message=(
                "`max_tuning_runs` must be non-negative, instead got {}.".format(
                    max_tuning_runs
                )
            ),
        )

    def fit_wandb(original, self, *args, **kwargs):
        """
        Autologging function that performs model training by executing the training method
        referred to be `func_name` on the instance of `clazz` referred to by `self` & records
        wandb parameters, metrics, tags, and artifacts to a corresponding wandb run.
        """
        # Obtain a copy of the training dataset prior to model training for subsequent
        # use during model logging & input example extraction, ensuring that we don't
        # attempt to infer input examples on data that was mutated during training
        (x, y_true, sample_weight) = _get_x_y_and_sample_weight(self.fit, args, kwargs)
        _log_pretraining_metadata(self, *args, **kwargs)

        fit_output = original(self, *args, **kwargs)
        _log_posttraining_metadata(self, x, y_true, sample_weight)

        return fit_output

    def _log_pretraining_metadata(
            estimator, *args, **kwargs
    ):  # pylint: disable=unused-argument
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        wandb run.

        param estimator: The scikit-learn estimator for which to log metadata.
        param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """
        # Deep parameter logging includes parameters from children of a given
        # estimator. For some meta estimators (e.g., pipelines), recording
        # these parameters is desirable. For parameter search estimators,
        # however, child estimators act as seeds for the parameter search
        # process; accordingly, we avoid logging initial, untuned parameters
        # for these seed estimators.
        should_log_params_deeply = not _is_parameter_search_estimator(estimator)

        run_id = wandb.run._run_id if wandb.run is not None else None
        if run_id:
            wandb.config.update(estimator.get_params(deep=should_log_params_deeply))

    def _log_posttraining_metadata(
            estimator, x, y, sample_weight,
    ):
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        wandb run .
.
        param estimator: The scikit-learn estimator for which to log metadata.
        param x: The training dataset samples passed to the ``estimator.fit()`` function.
        param y: The training dataset labels passed to the ``estimator.fit()`` function.
        param sample_weight: Sample weights passed to the ``estimator.fit()`` function.
        """
        # Fetch an input example using the first several rows of the array-like
        # training data supplied to the training routine (e.g., `fit()`). Copy the
        # example to avoid mutation during subsequent metric computations
        input_example_exc = None
        try:
            input_example = deepcopy(x[:INPUT_EXAMPLE_SAMPLE_ROWS])
        except Exception as e:
            input_example_exc = e

        def get_input_example():
            if input_example_exc is not None:
                raise input_example_exc
            else:
                return input_example

        # log common metrics and artifacts for estimators (classifier, regressor)
        logged_metrics = _log_estimator_content(
            estimator=estimator,
            prefix=_TRAINING_PREFIX,
            x=x,
            y_true=y,
            sample_weight=sample_weight,
            pos_label=pos_label,
        )
        if y is None and not logged_metrics:
            _logger.warning(
                "Training metrics will not be recorded because training labels were not specified."
                " To automatically record training metrics, provide training labels as inputs to"
                " the model training function."
            )

        def _log_model_with_except_handling(*args, **kwargs):
            try:
                return log_model(*args, **kwargs)
            except _SklearnCustomModelPicklingError as e:
                _logger.warning(str(e))

        if log_models:
            _log_model_with_except_handling(
                estimator,
                artifact_path=f"{estimator.__class__.__name__}_model",
                metrics=logged_metrics,
            )

        if _is_parameter_search_estimator(estimator):
            if hasattr(estimator, "best_estimator_") and log_models:
                _log_model_with_except_handling(
                    estimator.best_estimator_, artifact_path="best_estimator",
                )

            if hasattr(estimator, "best_score_"):
                wandb.log({"best_cv_score": estimator.best_score_})

            if hasattr(estimator, "best_params_"):
                best_params = {
                    "best_{param_name}".format(param_name=param_name): param_value
                    for param_name, param_value in estimator.best_params_.items()
                }
                wandb.config.update(best_params)

            if hasattr(estimator, "cv_results_"):
                try:
                    cv_results_df = pd.DataFrame.from_dict(estimator.cv_results_)
                    _log_parameter_search_results_as_artifact(cv_results_df, )
                except Exception as e:

                    msg = (
                        "Failed to log parameter search results as an artifact."
                        " Exception: {}".format(str(e))
                    )
                    _logger.warning(msg)

    def patched_fit(fit_impl, original, self, *args, **kwargs):
        """
        Autologging patch function to be applied to a sklearn model class that defines a `fit`
        method and inherits from `BaseEstimator` (thereby defining the `get_params()` method)
        """
        should_log_post_training_metrics = (
                log_post_training_metrics
                and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        )
        with _SklearnTrainingSession(clazz=self.__class__, allow_children=False) as t:
            if t.should_log():
                # In `fit_wandb` call, it will also call metric API for computing training metrics
                # so, we need temporarily disable the post_training_metrics patching.
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    result = fit_impl(original, self, *args, **kwargs)
                if should_log_post_training_metrics:
                    _AUTOLOGGING_METRICS_MANAGER.register_model(self, wandb.run._run_id)
                return result
            else:
                return original(self, *args, **kwargs)

    def patched_predict(original, self, *args, **kwargs):
        """
        In `patched_predict`, register the prediction result instance with the run id and
         eval dataset name. e.g.
        ```
        prediction_result = model_1.predict(eval_X)
        ```
        then we need register the following relationship into the `_AUTOLOGGING_METRICS_MANAGER`:
        id(prediction_result) --> (eval_dataset_name, run_id)

        Note: we cannot set additional attributes "eval_dataset_name" and "run_id" into
        the prediction_result object, because certain dataset type like numpy does not support
        additional attribute assignment.
        """
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            # Avoid nested patch when nested inference calls happens.
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                predict_result = original(self, *args, **kwargs)
            eval_dataset = _get_instance_method_first_arg_value(original, args, kwargs)
            eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(
                self, eval_dataset
            )
            _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(
                run_id, eval_dataset_name, predict_result
            )

            return predict_result
        else:
            return original(self, *args, **kwargs)

    def patched_metric_api(original, *args, **kwargs):
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
            # one metric api may call another metric api,
            # to avoid this, call disable_log_post_training_metrics to avoid nested patch
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                metric = original(*args, **kwargs)

            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
                metric_name = original.__name__
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(
                    None, original, *args, **kwargs
                )

                (
                    run_id,
                    dataset_name,
                ) = _AUTOLOGGING_METRICS_MANAGER.get_run_id_and_dataset_name_for_metric_api_call(
                    args, kwargs
                )
                if run_id and dataset_name:
                    metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(
                        run_id, metric_name, dataset_name, call_command
                    )
                    _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(
                        run_id, metric_key, metric
                    )

            return metric
        else:
            return original(*args, **kwargs)

    # we need patch model.score method because:
    #  some model.score() implementation won't call metric APIs in `sklearn.metrics`
    #  e.g.
    #  https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/covariance/_empirical_covariance.py#L220
    def patched_model_score(original, self, *args, **kwargs):
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            # `model.score` may call metric APIs internally, in order to prevent nested metric call
            # being logged, temporarily disable post_training_metrics patching.
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                score_value = original(self, *args, **kwargs)

            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(score_value):
                metric_name = f"{self.__class__.__name__}_score"
                call_command = _AUTOLOGGING_METRICS_MANAGER.gen_metric_call_command(
                    self, original, *args, **kwargs
                )

                eval_dataset = _get_instance_method_first_arg_value(
                    original, args, kwargs
                )
                eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(
                    self, eval_dataset
                )
                metric_key = _AUTOLOGGING_METRICS_MANAGER.register_metric_api_call(
                    run_id, metric_name, eval_dataset_name, call_command
                )
                _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(
                    run_id, metric_key, score_value
                )

            return score_value
        else:
            return original(self, *args, **kwargs)

    def _apply_sklearn_descriptor_unbound_method_call_fix():
        import sklearn

        if Version(sklearn.__version__) <= Version("0.24.2"):
            import sklearn.utils.metaestimators

            # pylint: disable=redefined-builtin,unused-argument
            def patched_IffHasAttrDescriptor__get__(self, obj, type=None):
                """
                For sklearn version <= 0.24.2, `_IffHasAttrDescriptor.__get__` method does not
                support unbound method call.
                See https://github.com/scikit-learn/scikit-learn/issues/20614
                This patched function is for hot patch.
                """

                # raise an AttributeError if the attribute is not present on the object
                if obj is not None:
                    # delegate only on instances, not the classes.
                    # this is to allow access to the docstrings.
                    for delegate_name in self.delegate_names:
                        try:
                            delegate = sklearn.utils.metaestimators.attrgetter(
                                delegate_name
                            )(obj)
                        except AttributeError:
                            continue
                        else:
                            getattr(delegate, self.attribute_name)
                            break
                    else:
                        sklearn.utils.metaestimators.attrgetter(
                            self.delegate_names[-1]
                        )(obj)

                    def out(*args, **kwargs):
                        return self.fn(obj, *args, **kwargs)

                else:
                    # This makes it possible to use the decorated method as an unbound method,
                    # for instance when monkeypatching.
                    def out(*args, **kwargs):
                        return self.fn(*args, **kwargs)

                # update the docstring of the returned function
                functools.update_wrapper(out, self.fn)
                return out

            update_wrapper_extended(
                patched_IffHasAttrDescriptor__get__,
                sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__,
            )

            sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__ = (
                patched_IffHasAttrDescriptor__get__
            )

    _apply_sklearn_descriptor_unbound_method_call_fix()

    estimators_to_patch = _gen_estimators_to_patch()
    patched_fit_impl = fit_wandb
    for class_def in estimators_to_patch:
        # Patch fitting methods
        for func_name in ["fit", "fit_transform", "fit_predict"]:
            _patch_estimator_method_if_available(
                flavour_name,
                class_def,
                func_name,
                functools.partial(patched_fit, patched_fit_impl),
            )

        # Patch inference methods
        for func_name in ["predict", "predict_proba", "transform", "predict_log_proba"]:
            _patch_estimator_method_if_available(
                flavour_name, class_def, func_name, patched_predict,
            )

            # Patch scoring methods
            _patch_estimator_method_if_available(
                flavour_name, class_def, "score", patched_model_score,
            )

    if log_post_training_metrics:
        for metric_name in _get_metric_name_list():
            safe_patch(
                flavour_name, sklearn.metrics, metric_name, patched_metric_api,
            )

        for scorer in sklearn.metrics.SCORERS.values():
            safe_patch(flavour_name, scorer, "_score_func", patched_metric_api)

    def patched_fn_with_autolog_disabled(original, *args, **kwargs):
        with disable_autologging():
            return original(*args, **kwargs)

    for disable_autolog_func_name in _apis_autologging_disabled:
        safe_patch(
            flavour_name,
            sklearn.model_selection,
            disable_autolog_func_name,
            patched_fn_with_autolog_disabled,
        )


EVAL_PREFIX = "eval_"


def eval_and_log_metrics(
        model, x, y_true, *, prefix="", sample_weight=None, pos_label=None
):
    """
    Computes and logs metrics (and artifacts) for the given model and labeled dataset.
    The metrics/artifacts mirror what is auto-logged when training a model.

    param model: The model to be evaluated.
    param x: The features for the evaluation dataset.
    param y_true: The labels for the evaluation dataset.
    param prefix: Prefix used to name metrics and artifacts.
    param sample_weight: Per-sample weights to apply in the computation of metrics/artifacts.
    param pos_label: The positive label used to compute binary classification metrics such as
        precision, recall, f1, etc. This parameter is only used for binary classification model
        - if used on multi-label model, the evaluation will fail;
        - if used for regression model, the parameter will be ignored.
        For multi-label classification, keep `pos_label` unset (or set to `None`), and the
        function will calculate metrics for each label and find their average weighted by support
        (number of true instances for each label).
    return: The dict of logged metrics. Artifacts can be retrieved by inspecting the run.




    Each metric's and artifact's name is prefixed with `prefix`, e.g., in the previous example the
    metrics and artifacts are named 'val_XXXXX'. Note that training-time metrics are auto-logged
    as 'training_XXXXX'. Metrics and artifacts are logged under the currently active run if one
    exists, otherwise a new run is started and left active.

    Raises an error if:
      - prefix is empty
      - model is not a sklearn estimator or does not support the 'predict' method
    """
    metrics_manager = _AUTOLOGGING_METRICS_MANAGER
    if not prefix:
        _logger.warning(
            f"The prefix cannot be empty. Setting prefix to '{EVAL_PREFIX}'."
        )
        prefix = EVAL_PREFIX
    with metrics_manager.disable_log_post_training_metrics():
        return _eval_and_log_metrics_impl(
            model,
            x,
            y_true,
            prefix=prefix,
            sample_weight=sample_weight,
            pos_label=pos_label,
        )


def _eval_and_log_metrics_impl(model, x, y_true, *, prefix, sample_weight, pos_label):
    from integration.sklearn.utils import _log_estimator_content
    from sklearn.base import BaseEstimator

    if prefix is None or prefix == "":
        raise ValueError("Must specify a non-empty prefix")

    if not isinstance(model, BaseEstimator):
        raise ValueError(
            "The provided model was not a sklearn estimator. Please ensure the passed-in model is "
            "a sklearn estimator subclassing sklearn.base.BaseEstimator"
        )

    if not hasattr(model, "predict"):
        raise ValueError(
            "Model does not support predictions. Please pass a model object defining a predict() "
            "method"
        )

    metrics = _log_estimator_content(
        estimator=model,
        prefix=prefix,
        x=x,
        y_true=y_true,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    return metrics
