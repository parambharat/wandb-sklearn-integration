import collections
import functools
import inspect
import logging
import os
import pickle
import shutil
import tempfile
from abc import abstractmethod
from copy import deepcopy

import numpy as np
from . import gorilla
from .autologging_utils import (
    _AutologgingSessionManager,
    _AUTOLOGGING_GLOBALLY_DISABLED,
    AutologgingEventLogger,
)
from packaging.version import Version

import wandb

_logger = logging.getLogger(__name__)

# The prefix to note that all calculated metrics and artifacts are solely based on training datasets
_TRAINING_PREFIX = "training_"

_SAMPLE_WEIGHT = "sample_weight"

# _SklearnArtifact represents an artifact (e.g. confusion matrix) that will be computed and
# logged during the autologging routine for a particular model type (eg, classifier, regressor).
_SklearnArtifact = collections.namedtuple(
    "_SklearnArtifact", ["name", "function", "arguments", "title"]
)

# _SklearnMetric represents a metric (e.g, precision_score) that will be computed and
# logged during the autologging routine for a particular model type (eg, classifier, regressor).
_SklearnMetric = collections.namedtuple(
    "_SklearnMetric", ["name", "function", "arguments"]
)
_AUTOLOGGING_PATCHES = {}


class TempDir:
    def __init__(self, chdir=False, remove_on_exit=True):
        self._dir = None
        self._path = None
        self._chdir = chdir
        self._remove = remove_on_exit

    def __enter__(self):
        self._path = os.path.abspath(tempfile.mkdtemp())
        assert os.path.exists(self._path)
        if self._chdir:
            self._dir = os.path.abspath(os.getcwd())
            os.chdir(self._path)
        return self

    def __exit__(self, tp, val, traceback):
        if self._chdir and self._dir:
            os.chdir(self._dir)
            self._dir = None
        if self._remove and os.path.exists(self._path):
            shutil.rmtree(self._path)

        assert not self._remove or not os.path.exists(self._path)
        assert os.path.exists(os.getcwd())

    def path(self, *path):
        return (
            os.path.join("/", *path) if self._chdir else os.path.join(self._path, *path)
        )


class _SklearnCustomModelPicklingError(pickle.PicklingError):
    """
    Exception for describing error raised during pickling custom sklearn estimator
    """

    def __init__(self, sk_model, original_exception):
        """
        param sk_model: The custom sklearn model to be pickled
        param original_exception: The original exception raised
        """
        super().__init__(
            f"Pickling custom sklearn model {sk_model.__class__.__name__} failed "
            f"when saving model: {str(original_exception)}"
        )
        self.original_exception = original_exception


def _inspect_original_var_name(var, fallback_name):
    """
    Inspect variable name, will search above frames and fetch the same instance variable name
    in the most outer frame.
    If inspect failed, return fallback_name
    """
    import inspect

    if var is None:
        return fallback_name
    try:
        original_var_name = fallback_name

        frame = inspect.currentframe().f_back
        while frame is not None:
            arg_info = inspect.getargvalues(frame)  # pylint: disable=deprecated-method

            fixed_args = [arg_info.locals[arg_name] for arg_name in arg_info.args]
            varlen_args = (
                list(arg_info.locals[arg_info.varargs]) if arg_info.varargs else []
            )
            keyword_args = (
                list(arg_info.locals[arg_info.keywords].values())
                if arg_info.keywords
                else []
            )

            all_args = fixed_args + varlen_args + keyword_args

            # check whether `var` is in arg list first. If yes, go to check parent frame.
            if any(var is arg for arg in all_args):
                # the var is passed in from caller, check parent frame.
                frame = frame.f_back
                continue

            for var_name, var_val in frame.f_locals.items():
                if var_val is var:
                    original_var_name = var_name
                    break

            break

        return original_var_name

    except Exception:
        return fallback_name


def update_wrapper_extended(wrapper, wrapped):
    """
    Update a `wrapper` function to look like the `wrapped` function. This is an extension of
    `functools.update_wrapper` that applies the docstring *and* signature of `wrapped` to
    `wrapper`, producing a new function.

    :return: A new function with the same implementation as `wrapper` and the same docstring
             & signature as `wrapped`.
    """
    updated_wrapper = functools.update_wrapper(wrapper, wrapped)
    # Assign the signature of the `wrapped` function to the updated wrapper function.
    # Certain frameworks may disallow signature inspection, causing `inspect.signature()` to throw.
    try:
        updated_wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:
        _logger.debug(
            "Failed to restore original signature for wrapper around %s", wrapped
        )
    return updated_wrapper


def _backported_all_estimators(type_filter=None):
    """
    Backported from scikit-learn 0.23.2:
    https://github.com/scikit-learn/scikit-learn/blob/0.23.2/sklearn/utils/__init__.py#L1146

    Use this backported `all_estimators` in old versions of sklearn because:
    1. An inferior version of `all_estimators` that old versions of sklearn use for testing,
       might function differently from a newer version.
    2. This backported `all_estimators` works on old versions of sklearn that don’t even define
       the testing utility variant of `all_estimators`.

    ========== original docstring ==========
    Get a list of all estimators from sklearn.
    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default, meta_estimators such as GridSearchCV are also not included.
    Parameters
    ----------
    type_filter : string, list of string,  or None, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.
    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    # lazy import to avoid circular imports from sklearn.base
    import pkgutil
    import platform
    import sklearn
    from importlib import import_module
    from operator import itemgetter

    # pylint: disable=no-name-in-module, import-error
    from sklearn.utils._testing import ignore_warnings
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
        TransformerMixin,
        ClusterMixin,
    )

    IS_PYPY = platform.python_implementation() == "PyPy"

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests", "externals", "setup", "conftest"}
    root = sklearn.__path__[0]  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, modname, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            # TODO: Remove when FeatureHasher is implemented in PYPY
            # Skips FeatureHasher for PYPY
            if IS_PYPY and "feature_extraction" in modname:
                classes = [
                    (name, est_cls)
                    for name, est_cls in classes
                    if name == "FeatureHasher"
                ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                " %s." % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def _gen_estimators_to_patch():
    _, estimators_to_patch = zip(*_all_estimators())
    # Ensure that relevant meta estimators (e.g. GridSearchCV, Pipeline) are selected
    # for patching if they are not already included in the output of `all_estimators()`
    estimators_to_patch = set(estimators_to_patch).union(
        set(_get_meta_estimators_for_autologging())
    )
    # Exclude certain preprocessing & feature manipulation estimators from patching. These
    # estimators represent data manipulation routines (e.g., normalization, label encoding)
    # rather than ML algorithms. Accordingly, we should not create wandb runs and log
    # parameters / metrics for these routines, unless they are captured as part of an ML pipeline
    # (via `sklearn.pipeline.Pipeline`)
    excluded_module_names = [
        "sklearn.preprocessing",
        "sklearn.impute",
        "sklearn.feature_extraction",
        "sklearn.feature_selection",
    ]

    excluded_class_names = [
        "sklearn.compose._column_transformer.ColumnTransformer",
    ]

    return [
        estimator
        for estimator in estimators_to_patch
        if not any(
            estimator.__module__.startswith(excluded_module_name)
            or (estimator.__module__ + "." + estimator.__name__) in excluded_class_names
            for excluded_module_name in excluded_module_names
        )
    ]


_metric_api_excluding_list = ["check_scoring", "get_scorer", "make_scorer"]


def _get_metric_name_list():
    """
    Return metric function name list in `sklearn.metrics` module
    """
    from sklearn import metrics

    metric_list = []
    for metric_method_name in metrics.__all__:
        # excludes plot_* methods
        # exclude class (e.g. metrics.ConfusionMatrixDisplay)
        metric_method = getattr(metrics, metric_method_name)
        if (
            metric_method_name not in _metric_api_excluding_list
            and not inspect.isclass(metric_method)
            and callable(metric_method)
            and not metric_method_name.startswith("plot_")
        ):
            metric_list.append(metric_method_name)
    return metric_list


def _all_estimators():
    try:
        from sklearn.utils import all_estimators

        return all_estimators()
    except ImportError:
        return _backported_all_estimators()


def _get_estimator_info_tags(estimator):
    """
    Get estimator info and tags from an estimator instance.
    :param estimator: An sklearn estimator instance
    :return: A dictionary of wandb run tag keys and values
             describing the specified estimator.
    """
    return (
        {
            "estimator_name": estimator.__class__.__name__,
            "estimator_class": (
                estimator.__class__.__module__ + "." + estimator.__class__.__name__
            ),
        },
    )


def _get_meta_estimators_for_autologging():
    """
    :return: A list of meta estimator class definitions
             (e.g., `sklearn.model_selection.GridSearchCV`) that should be included
             when patching training functions for autologging
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    return [
        GridSearchCV,
        RandomizedSearchCV,
        Pipeline,
    ]


def _is_parameter_search_estimator(estimator):
    """
    Given an estimator, return `True` if the estimator is a parameter search estimator
    :param estimator: An sklearn estimator instance
    :return: `True` if the specified scikit-learn estimator is a parameter search estimator,
             such as `GridSearchCV`. `False` otherwise.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    parameter_search_estimators = [
        GridSearchCV,
        RandomizedSearchCV,
    ]

    return any(
        isinstance(estimator, param_search_estimator)
        for param_search_estimator in parameter_search_estimators
    )


# Util function to check whether a metric is can be computed in given sklearn version
def _is_metric_supported(metric_name):
    import sklearn

    # This dict can be extended to store special metrics' specific supported versions
    _metric_supported_version = {"roc_auc_score": "0.22.2"}

    return Version(sklearn.__version__) >= Version(
        _metric_supported_version[metric_name]
    )


def _is_metric_value_loggable(metric_value):
    """
    check whether the specified `metric_value` is a numeric value which can be logged
    as a wandb metric.
    """
    return isinstance(metric_value, (int, float, np.number)) and not isinstance(
        metric_value, bool
    )


# Util function to check whether artifact plotting functions can be computed
# in given sklearn version (should >= 0.22.0)
def _is_plotting_supported():
    import sklearn

    return Version(sklearn.__version__) >= Version("0.22.0")


def _get_class_labels_from_estimator(estimator):
    """
    Extracts class labels from `estimator` if `estimator.classes` is available.
    """
    return estimator.classes_ if hasattr(estimator, "classes_") else None


def _log_warning_for_metrics(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The metric "
        + func_name
        + " will not be recorded."
        + " Metric error: "
        + str(err)
    )
    _logger.warning(msg)


def _log_warning_for_artifacts(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The artifact "
        + func_name
        + " will not be recorded."
        + " Artifact error: "
        + str(err)
    )
    _logger.warning(msg)


def _get_metrics_value_dict(metrics_list):
    metric_value_dict = {}
    for metric in metrics_list:
        try:
            metric_value = metric.function(**metric.arguments)
        except Exception as e:
            _log_warning_for_metrics(metric.name, metric.function, e)
        else:
            metric_value_dict[metric.name] = metric_value
    return metric_value_dict


def _get_classifier_metrics(
    fitted_estimator, prefix, x, y_true, sample_weight, pos_label
):
    """
    Compute and record various common metrics for classifiers

    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, when `pos_label` is not specified (passed in as `None`), we set `average`
    to `weighted` to compute the weighted score of these metrics.
    When the `pos_label` is specified (not `None`), we set `average` to `binary`.

    For (4) accuracy score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy,
    as opposed to `False` that outputs the absolute correct number of sample prediction

    We log additional metrics if certain classifier has method `predict_proba`
    (5) log loss:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    (6) roc_auc_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    By default, for roc_auc_score, we pick `average` to be `weighted`, `multi_class` to be `ovo`,
    to make the output more insensitive to dataset imbalance.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, ...... sample_weight), otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)

    :param fitted_estimator: The already fitted classifier

    """
    import sklearn

    average = "weighted" if pos_label is None else "binary"
    y_pred = fitted_estimator.predict(x)

    classifier_metrics = [
        _SklearnMetric(
            name=prefix + "precision_score",
            function=sklearn.metrics.precision_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "recall_score",
            function=sklearn.metrics.recall_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "f1_score",
            function=sklearn.metrics.f1_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        ),
        _SklearnMetric(
            name=prefix + "accuracy_score",
            function=sklearn.metrics.accuracy_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                normalize=True,
                sample_weight=sample_weight,
            ),
        ),
    ]

    if hasattr(fitted_estimator, "predict_proba"):
        y_pred_proba = fitted_estimator.predict_proba(x)
        classifier_metrics.extend(
            [
                _SklearnMetric(
                    name=prefix + "log_loss",
                    function=sklearn.metrics.log_loss,
                    arguments=dict(
                        y_true=y_true, y_pred=y_pred_proba, sample_weight=sample_weight
                    ),
                ),
            ]
        )

        if _is_metric_supported("roc_auc_score"):
            # For binary case, the parameter `y_score` expect scores must be
            # the scores of the class with the greater label.
            if len(y_pred_proba[0]) == 2:
                y_pred_proba = y_pred_proba[:, 1]

            classifier_metrics.extend(
                [
                    _SklearnMetric(
                        name=prefix + "roc_auc_score",
                        function=sklearn.metrics.roc_auc_score,
                        arguments=dict(
                            y_true=y_true,
                            y_score=y_pred_proba,
                            average="weighted",
                            sample_weight=sample_weight,
                            multi_class="ovo",
                        ),
                    ),
                ]
            )

    return _get_metrics_value_dict(classifier_metrics)


def _get_regressor_metrics(fitted_estimator, prefix, x, y_true, sample_weight):
    """
    Compute and record various common metrics for regressors

    For (1) (root) mean squared error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    (2) mean absolute error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    (3) r2 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    By default, we choose the parameter `multioutput` to be `uniform_average`
    to average outputs with uniform weight.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a dictionary of metric(name, value)

    :param fitted_estimator: The already fitted regressor
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    y_pred = fitted_estimator.predict(x)

    regressor_metrics = [
        _SklearnMetric(
            name=prefix + "mse",
            function=sklearn.metrics.mean_squared_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=prefix + "mae",
            function=sklearn.metrics.mean_absolute_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=prefix + "r2_score",
            function=sklearn.metrics.r2_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
    ]

    # To be compatible with older versions of scikit-learn (below 0.22.2), where
    # `sklearn.metrics.mean_squared_error` does not have "squared" parameter to calculate `rmse`,
    # we compute it through np.sqrt(<value of mse>)
    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[prefix + "rmse"] = np.sqrt(metrics_value_dict[prefix + "mse"])

    return metrics_value_dict


## TODO: (FIXME): Change to use wandb.sklearn.plot utils
def _get_classifier_artifacts(fitted_estimator, prefix, x, y_true, sample_weight):
    """
    Draw and record various common artifacts for classifier

    For all classifiers, we always log:
    (1) confusion matrix:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

    For only binary classifiers, we will log:
    (2) precision recall curve:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html
    (3) roc curve:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and split into train & test datasets.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a list of artifacts path to be logged

    param fitted_estimator: The already fitted regressor
    :return: List of artifacts to be logged
    """
    import sklearn

    if not _is_plotting_supported():
        return []

    is_plot_function_deprecated = Version(sklearn.__version__) >= Version("1.0")

    def plot_confusion_matrix(*args, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt

        class_labels = _get_class_labels_from_estimator(fitted_estimator)
        if class_labels is None:
            class_labels = set(y_true)

        with matplotlib.rc_context(
            {
                "font.size": min(8.0, 50.0 / len(class_labels)),
                "axes.labelsize": 8.0,
                "figure.dpi": 175,
            }
        ):
            _, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
            return (
                sklearn.metrics.ConfusionMatrixDisplay.from_estimator(
                    *args, **kwargs, ax=ax
                )
                if is_plot_function_deprecated
                else sklearn.metrics.plot_confusion_matrix(*args, **kwargs, ax=ax)
            )

    y_true_arg_name = "y" if is_plot_function_deprecated else "y_true"
    classifier_artifacts = [
        _SklearnArtifact(
            name=prefix + "confusion_matrix",
            function=plot_confusion_matrix,
            arguments=dict(
                estimator=fitted_estimator,
                X=x,
                sample_weight=sample_weight,
                normalize="true",
                cmap="Blues",
                **{y_true_arg_name: y_true},
            ),
            title="Normalized confusion matrix",
        ),
    ]

    # The plot_roc_curve and plot_precision_recall_curve can only be
    # supported for binary classifier
    if len(set(y_true)) == 2:
        classifier_artifacts.extend(
            [
                _SklearnArtifact(
                    name=prefix + "roc_curve",
                    function=sklearn.metrics.RocCurveDisplay.from_estimator
                    if is_plot_function_deprecated
                    else sklearn.metrics.plot_roc_curve,
                    arguments=dict(
                        estimator=fitted_estimator,
                        X=x,
                        y=y_true,
                        sample_weight=sample_weight,
                    ),
                    title="ROC curve",
                ),
                _SklearnArtifact(
                    name=prefix + "precision_recall_curve",
                    function=sklearn.metrics.PrecisionRecallDisplay.from_estimator
                    if is_plot_function_deprecated
                    else sklearn.metrics.plot_precision_recall_curve,
                    arguments=dict(
                        estimator=fitted_estimator,
                        X=x,
                        y=y_true,
                        sample_weight=sample_weight,
                    ),
                    title="Precision recall curve",
                ),
            ]
        )

    return classifier_artifacts


def _log_specialized_estimator_content(
    fitted_estimator, prefix, x, y_true, sample_weight, pos_label
):
    import sklearn

    metrics = dict()

    if y_true is not None:
        try:
            if sklearn.base.is_classifier(fitted_estimator):
                metrics = _get_classifier_metrics(
                    fitted_estimator, prefix, x, y_true, sample_weight, pos_label
                )
            elif sklearn.base.is_regressor(fitted_estimator):
                metrics = _get_regressor_metrics(
                    fitted_estimator, prefix, x, y_true, sample_weight
                )
        except Exception as err:
            msg = (
                "Failed to autolog metrics for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(err)
            )
            _logger.warning(msg)
        else:
            wandb.log(metrics)

    if sklearn.base.is_classifier(fitted_estimator):
        try:
            artifacts = _get_classifier_artifacts(
                fitted_estimator, prefix, x, y_true, sample_weight
            )
        except Exception as e:
            msg = (
                "Failed to autolog artifacts for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(e)
            )
            _logger.warning(msg)
            return metrics

        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as ie:
            _logger.warning(
                f"Failed to import matplotlib (error: {repr(ie)}). Skipping artifact logging."
            )
            return metrics

        _matplotlib_config = {
            "savefig.dpi": 175,
            "figure.autolayout": True,
            "font.size": 8,
        }
        ## TODO: Fix this. This is a hack to get around the fact that wandb.log_artifacts cannot use a temp path
        with TempDir() as tmp_dir:
            for artifact in artifacts:
                try:
                    with matplotlib.rc_context(_matplotlib_config):
                        display = artifact.function(**artifact.arguments)
                        display.ax_.set_title(artifact.title)
                        artifact_path = "{}.png".format(artifact.name)
                        filepath = tmp_dir.path(artifact_path)
                        display.figure_.savefig(fname=filepath, format="png")
                        wandb.log({f"{artifact.name}": display.figure_})
                        plt.close(display.figure_)
                except Exception as e:
                    _log_warning_for_artifacts(artifact.name, artifact.function, e)
            if artifacts:
                wandb_artifact = wandb.Artifact(
                    f"{fitted_estimator.__class__.__name__}_plots",
                    type="plots",
                    metadata=_matplotlib_config,
                )
                wandb_artifact.add_dir(tmp_dir.path())
                wandb.log_artifact(wandb_artifact)

    return metrics


def _log_estimator_content(
    estimator, prefix, x, y_true=None, sample_weight=None, pos_label=None,
):
    """
    Logs content for the given estimator, which includes metrics and artifacts that might be
    tailored to the estimator's type (e.g., regression vs classification). Training labels
    are required for metric computation; metrics will be omitted if labels are not available.

    param estimator: The estimator used to compute metrics and artifacts.
    param prefix: A prefix used to name the logged content. Typically, it's 'training_' for
                   training-time content and user-controlled for evaluation-time content.
    param x: The data feature samples.
    param y_true: Labels.
    :param sample_weight: Per-sample weights used in the computation of metrics and artifacts.
    param pos_label: The positive label used to compute binary classification metrics such as
        precision, recall, f1, etc. This parameter is only used for classification metrics.
        If set to `None`, the function will calculate metrics for each label and find their
        average weighted by support (number of true instances for each label).
    :return: A dict of the computed metrics.
    """
    metrics = _log_specialized_estimator_content(
        fitted_estimator=estimator,
        prefix=prefix,
        x=x,
        y_true=y_true,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    if hasattr(estimator, "score") and y_true is not None:
        try:
            # Use the sample weight only if it is present in the score args
            score_arg_names = _get_arg_names(estimator.score)
            score_args = (
                (x, y_true, sample_weight)
                if _SAMPLE_WEIGHT in score_arg_names
                else (x, y_true)
            )
            score = estimator.score(*score_args)
        except Exception as e:
            msg = (
                estimator.score.__qualname__
                + " failed. The 'training_score' metric will not be recorded. Scoring error: "
                + str(e)
            )
            _logger.warning(msg)
        else:
            score_key = prefix + "score"
            wandb.log({score_key: score})
            metrics[score_key] = score

    return metrics


def _log_parameter_search_results_as_artifact(cv_results_df,):
    """
    Records a collection of parameter search results as a wandb artifact
    for the specified run.

    param cv_results_df: A Pandas DataFrame containing the results of a parameter search
                          training session, which may be obtained by parsing the `cv_results_`
                          attribute of a trained parameter search estimator such as
                          `GridSearchCV`.
    """
    with TempDir() as t:
        results_path = t.path("cv_results.csv")
        cv_results_df.to_csv(results_path, index=False)
        wandb_artifact = wandb.Artifact("cv_results", type="parameter_search_results")
        wandb_artifact.add_file(results_path)
        wandb.log_artifact(wandb_artifact)
        wandb.log({"cv_results": wandb.Table(dataframe=cv_results_df)})


def log_model(sk_model, artifact_path, **kwargs):
    """
    Log a scikit-learn model as an artifact for the current run.

    param sk_model: The scikit-learn model to be saved and logged.
    param artifact_path: The run-relative artifact path in which to save the model.
    """
    import sklearn
    import joblib

    if not isinstance(sk_model, sklearn.base.BaseEstimator):
        raise TypeError(
            "Invalid model type: '{model_type}'. Please provide a scikit-learn model"
            " that subclasses sklearn.base.BaseEstimator".format(
                model_type=type(sk_model)
            )
        )
    model_metadata = _get_estimator_info_tags(estimator=sk_model)[0]
    metrics = kwargs.get("metrics", None)
    if metrics is not None:
        model_metadata.update(metrics)

    with TempDir() as t:
        model_path = t.path(f"{artifact_path}.pkl")
        joblib.dump(sk_model, model_path)
        artifact = wandb.Artifact(artifact_path, type="model", metadata=model_metadata,)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


def _get_instance_method_first_arg_value(method, call_pos_args, call_kwargs):
    """
    Get instance method first argument value (exclude the `self` argument).
    :param method A `cls.method` object which includes the `self` argument.
    param call_pos_args: positional arguments excluding the first `self` argument.
    param call_kwargs: keywords arguments.
    """
    if len(call_pos_args) >= 1:
        return call_pos_args[0]
    else:
        param_sig = inspect.signature(method).parameters
        first_arg_name = list(param_sig.keys())[1]
        assert param_sig[first_arg_name].kind not in [
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ]
        return call_kwargs.get(first_arg_name)


def _get_arg_names(f):
    """
    Get the argument names of a function.

    param f: A function.
    :return: A list of argument names.
    """
    # `inspect.getargspec` or `inspect.getfullargspec` doesn't work properly for a wrapped function.
    # See https://hynek.me/articles/decorators#mangled-signatures for details.
    return list(inspect.signature(f).parameters.keys())


def _get_x_y_and_sample_weight(fit_func, fit_args, fit_kwargs):
    """
    Get a tuple of (x, y, sample_weight) in the following steps.

    1. Extract x and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func,
       extract it from fit_args or fit_kwargs and return (x, y, sample_weight),
       otherwise return (x, y)

    :param fit_func: A fit function object.
    param fit_args: Positional arguments given to fit_func.
    param fit_kwargs: Keyword arguments given to fit_func.

    :returns: A tuple of either (x, y, sample_weight), where `y` and `sample_weight` may be
              `None` if the specified `fit_args` and `fit_kwargs` do not specify labels or
              a sample weighting. Copies of `x` and `y` are made in order to avoid mutation
              of the dataset during training.
    """

    def _get_xy(args, kwargs, x_var_name, y_var_name):
        # corresponds to: model.fit(x, y)
        if len(args) >= 2:
            return args[:2]

        # corresponds to: model.fit(x, <y_var_name>=y)
        if len(args) == 1:
            return args[0], kwargs.get(y_var_name)

        # corresponds to: model.fit(<x_var_name>=x, <y_var_name>=y)
        return kwargs[x_var_name], kwargs.get(y_var_name)

    def _get_sample_weight(arg_names, args, kwargs):
        sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)

        # corresponds to: model.fit(x, y, ..., sample_weight)
        if len(args) > sample_weight_index:
            return args[sample_weight_index]

        # corresponds to: model.fit(x, y, ..., sample_weight=sample_weight)
        if _SAMPLE_WEIGHT in kwargs:
            return kwargs[_SAMPLE_WEIGHT]

        return None

    fit_arg_names = _get_arg_names(fit_func)
    # In most cases, x_var_name and y_var_name become "x" and "y", respectively.
    # However, certain sklearn models use different variable names for x and y.
    # E.g., see: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.fit
    x_var_name, y_var_name = fit_arg_names[:2]
    x, y = _get_xy(fit_args, fit_kwargs, x_var_name, y_var_name)
    if x is not None:
        x = deepcopy(x)
    if y is not None:
        y = deepcopy(y)
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if (_SAMPLE_WEIGHT in fit_arg_names)
        else None
    )

    return x, y, sample_weight


def _wrap_patch(destination, name, patch_obj, settings=None):
    """
    Apply a patch.

    param destination: Patch destination
    :param name: Name of the attribute at the destination
    :param patch_obj: Patch object, it should be a function or a property decorated function
                      to be assigned to the patch point {destination}.{name}
    :param settings: Settings for gorilla.Patch
    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    patch = gorilla.Patch(destination, name, patch_obj, settings=settings)
    gorilla.apply(patch)
    return patch


# def safe_patch(
#         destination, function_name, patch_function,
# ):
#     assert callable(patch_function)
#     original_fn = gorilla.get_original_attribute(
#         destination, function_name, bypass_descriptor_protocol=False
#     )
#     raw_original_obj = gorilla.get_original_attribute(
#         destination, function_name, bypass_descriptor_protocol=True
#     )
#     if original_fn != raw_original_obj:
#         raise RuntimeError(f"Unsupported patch on {str(destination)}.{function_name}")
#     else:
#         original = original_fn
#
#     def safe_patch_function(*args, **kwargs):
#         def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
#             try:
#                 original_fn_result = original_fn(*og_args, **og_kwargs)
#                 return original_fn_result
#             except Exception as original_fn_e:
#                 raise original_fn_e
#
#         try:
#
#             def call_original(*og_args, **og_kwargs):
#                 def _original_fn(*_og_args, **_og_kwargs):
#                     original_result = original(*_og_args, **_og_kwargs)
#                     return original_result
#
#                 return call_original_fn_with_event_logging(
#                     _original_fn, og_args, og_kwargs
#                 )
#
#             # Apply the name, docstring, and signature of `original` to `call_original`.
#             # This is important because several autologging patch implementations inspect
#             # the signature of the `original` argument during execution
#             call_original = update_wrapper_extended(call_original, original)
#
#             patch_function(call_original, *args, **kwargs)
#         except Exception as e:
#             raise e
#
#     safe_patch_obj = update_wrapper_extended(safe_patch_function, original)
#     new_patch = _wrap_patch(destination, function_name, safe_patch_obj)


class PatchFunction:
    """
    Base class representing a function patch implementation with a callback for error handling.
    `PatchFunction` should be subclassed and used in conjunction with `safe_patch` to
    safely modify the implementation of a function. Subclasses of `PatchFunction` should
    use `_patch_implementation` to define modified ("patched") function implementations and
    `_on_exception` to define cleanup logic when `_patch_implementation` terminates due
    to an unhandled exception.
    """

    @abstractmethod
    def _patch_implementation(self, original, *args, **kwargs):
        """
        Invokes the patch function code.

        param original: The original, underlying function over which the `PatchFunction`
                         is being applied.
        :param *args: The positional arguments passed to the original function.
        :param **kwargs: The keyword arguments passed to the original function.
        """
        pass

    @abstractmethod
    def _on_exception(self, exception):
        """
        Called when an unhandled standard Python exception (i.e. an exception inheriting from
        `Exception`) or a `KeyboardInterrupt` prematurely terminates the execution of
        `_patch_implementation`.

        :param exception: The unhandled exception thrown by `_patch_implementation`.
        """
        pass

    @classmethod
    def call(cls, original, *args, **kwargs):
        return cls().__call__(original, *args, **kwargs)

    def __call__(self, original, *args, **kwargs):
        try:
            return self._patch_implementation(original, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            try:
                self._on_exception(e)
            finally:
                # Regardless of what happens during the `_on_exception` callback, reraise
                # the original implementation exception once the callback completes
                raise e


def _store_patch(autologging_integration, patch):
    """
    Stores a patch for a specified autologging_integration class. Later to be used for being able
    to revert the patch when disabling autologging.

    param autologging_integration: The name of the autologging wandb_sklearn_integration associated with the
                                    patch.
    param patch: The patch to be stored.
    """
    if autologging_integration in _AUTOLOGGING_PATCHES:
        _AUTOLOGGING_PATCHES[autologging_integration].add(patch)
    else:
        _AUTOLOGGING_PATCHES[autologging_integration] = {patch}


def safe_patch(
    autologging_integration,
    destination,
    function_name,
    patch_function,
    # manage_run=False,
):
    """
    Patches the specified `function_name` on the specified `destination` class for autologging
    purposes, preceding its implementation with an error-safe copy of the specified patch
    `patch_function` with the following error handling behavior:
        - Exceptions thrown from the underlying / original function
          (`<destination>.<function_name>`) are propagated to the caller.
        - Exceptions thrown from other parts of the patched implementation (`patch_function`)
          are caught and logged as warnings.
    param autologging_integration: The name of the autologging wandb_sklearn_integration associated with the
                                    patch.
    param destination: The Python class on which the patch is being defined.
    param function_name: The name of the function to patch on the specified `destination` class.
    param patch_function: The patched function code to apply. This is either a `PatchFunction`
                           class definition or a function object. If it is a function object, the
                           first argument should be reserved for an `original` method argument
                           representing the underlying / original function. Subsequent arguments
                           should be identical to those of the original function being patched.
    """
    from integration.sklearn.autologging_utils import (
        get_autologging_config,
        autologging_is_disabled,
    )

    patch_is_class = inspect.isclass(patch_function)
    if patch_is_class:
        assert issubclass(patch_function, PatchFunction)
    else:
        assert callable(patch_function)

    original_fn = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=True
    )
    if original_fn != raw_original_obj:
        raise RuntimeError(f"Unsupport patch on {str(destination)}.{function_name}")
    elif isinstance(original_fn, property):
        is_property_method = True

        # For property decorated methods (a kind of method delegation), e.g.
        # class A:
        #   @property
        #   def f1(self):
        #     ...
        #     return delegated_f1
        #
        # suppose `a1` is an instance of class `A`,
        # `A.f1.fget` will get the original `def f1(self)` method,
        # and `A.f1.fget(a1)` will be equivalent to `a1.f1()` and
        # its return value will be the `delegated_f1` function.
        # So using the `property.fget` we can construct the (delegated) "original_fn"
        def original(self, *args, **kwargs):
            # the `original_fn.fget` will get the original method decorated by `property`
            # the `original_fn.fget(self)` will get the delegated function returned by the
            # property decorated method.
            bound_delegate_method = original_fn.fget(self)
            return bound_delegate_method(*args, **kwargs)

    else:
        original = original_fn
        is_property_method = False

    def safe_patch_function(*args, **kwargs):
        """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether it has been executed and whether it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """
        is_silent_mode = get_autologging_config(
            autologging_integration, "silent", False
        )

        # Whether to exclude autologged content from user-created wandb runs
        # (i.e. runs created manually via `wandb.init`)
        exclusive = get_autologging_config(autologging_integration, "exclusive", False)
        user_created_wandb_run_is_active = (
            wandb.run is not None and not _AutologgingSessionManager.active_session()
        )
        active_session_failed = (
            _AutologgingSessionManager.active_session() is not None
            and _AutologgingSessionManager.active_session().state == "failed"
        )

        if (
            active_session_failed
            or autologging_is_disabled(autologging_integration)
            or (user_created_wandb_run_is_active and exclusive)
            or _AUTOLOGGING_GLOBALLY_DISABLED
        ):
            return original(*args, **kwargs)

        # Whether the original / underlying function has been called during the
        # execution of patched code
        original_has_been_called = False
        # The value returned by the call to the original / underlying function during
        # the execution of patched code
        original_result = None
        # Whether an exception was raised from within the original / underlying function
        # during the execution of patched code
        failed_during_original = False
        # The active wandb run (if any) associated with patch code execution
        patch_function_run_for_testing = None
        # The exception raised during executing patching function
        patch_function_exception = None

        def try_log_autologging_event(log_fn, *args):
            try:
                log_fn(*args)
            except Exception as e:
                _logger.debug(
                    "Failed to log autologging event via '%s'. Exception: %s",
                    log_fn,
                    e,
                )

        def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
            try:
                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_start,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                )
                original_fn_result = original_fn(*og_args, **og_kwargs)

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_success,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                )
                return original_fn_result
            except Exception as original_fn_e:
                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_original_function_error,
                    session,
                    destination,
                    function_name,
                    og_args,
                    og_kwargs,
                    original_fn_e,
                )

                nonlocal failed_during_original
                failed_during_original = True
                raise

        with _AutologgingSessionManager.start_session(
            autologging_integration
        ) as session:
            try:

                def call_original(*og_args, **og_kwargs):
                    def _original_fn(*_og_args, **_og_kwargs):
                        nonlocal original_has_been_called
                        original_has_been_called = True

                        nonlocal original_result

                        original_result = original(*_og_args, **_og_kwargs)
                        return original_result

                    return call_original_fn_with_event_logging(
                        _original_fn, og_args, og_kwargs
                    )

                # Apply the name, docstring, and signature of `original` to `call_original`.
                # This is important because several autologging patch implementations inspect
                # the signature of the `original` argument during execution
                call_original = update_wrapper_extended(call_original, original)

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_patch_function_start,
                    session,
                    destination,
                    function_name,
                    args,
                    kwargs,
                )

                if patch_is_class:
                    patch_function.call(call_original, *args, **kwargs)
                else:
                    patch_function(call_original, *args, **kwargs)

                session.state = "succeeded"

                try_log_autologging_event(
                    AutologgingEventLogger.get_logger().log_patch_function_success,
                    session,
                    destination,
                    function_name,
                    args,
                    kwargs,
                )
            except Exception as e:
                session.state = "failed"
                patch_function_exception = e
                # Exceptions thrown during execution of the original function should be
                # propagated to the caller. Additionally, exceptions encountered during test
                # mode should be reraised to detect bugs in autologging implementations
                if failed_during_original:  # or is_testing():
                    raise

            try:
                if original_has_been_called:
                    return original_result
                else:
                    return call_original_fn_with_event_logging(original, args, kwargs)
            finally:
                # If original function succeeds, but `patch_function_exception` exists,
                # it represents patching code unexpected failure, so we call
                # `log_patch_function_error` in this case.
                # If original function failed, we don't call `log_patch_function_error`
                # even if `patch_function_exception` exists, because original function failure
                # means there's some error in user code (e.g. user provide wrong arguments)
                if patch_function_exception is not None and not failed_during_original:
                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_patch_function_error,
                        session,
                        destination,
                        function_name,
                        args,
                        kwargs,
                        patch_function_exception,
                    )

                    _logger.warning(
                        "Encountered unexpected error during %s autologging: %s",
                        autologging_integration,
                        patch_function_exception,
                    )

    if is_property_method:
        # Create a patched function (also property decorated)
        # like:
        #
        # class A:
        # @property
        # def get_bound_safe_patch_fn(self):
        #   original_fn.fget(self) # do availability check
        #   return bound_safe_patch_fn
        #
        # Suppose `a1` is instance of class A,
        # then `a1.get_bound_safe_patch_fn(*args, **kwargs)` will be equivalent to
        # `bound_safe_patch_fn(*args, **kwargs)`
        def get_bound_safe_patch_fn(self):
            # This `original_fn.fget` call is for availability check, if it raises error
            # then `hasattr(obj, {func_name})` will return False,
            # so it mimic the original property behavior.
            original_fn.fget(self)

            def bound_safe_patch_fn(*args, **kwargs):
                return safe_patch_function(self, *args, **kwargs)

            # Make bound method `instance.target_method` keep the same doc and signature
            bound_safe_patch_fn = update_wrapper_extended(
                bound_safe_patch_fn, original_fn.fget
            )
            # Here return the bound safe patch function because user call property decorated
            # method will like `instance.property_decorated_method(...)`, and internally it will
            # call the `bound_safe_patch_fn`, the argument list don't include the `self` argument,
            # so return bound function here.
            return bound_safe_patch_fn

        # Make unbound method `class.target_method` keep the same doc and signature
        get_bound_safe_patch_fn = update_wrapper_extended(
            get_bound_safe_patch_fn, original_fn.fget
        )
        safe_patch_obj = property(get_bound_safe_patch_fn)
    else:
        safe_patch_obj = update_wrapper_extended(safe_patch_function, original)

    new_patch = _wrap_patch(destination, function_name, safe_patch_obj)
    _store_patch(autologging_integration, new_patch)


def _patch_estimator_method_if_available(
    flavour_name, class_def, func_name, patched_fn
):
    if not hasattr(class_def, func_name):
        return

    original = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        class_def, func_name, bypass_descriptor_protocol=True
    )
    if raw_original_obj == original and (
        callable(original) or isinstance(original, property)
    ):
        # normal method or property decorated method
        safe_patch(
            flavour_name, class_def, func_name, patched_fn,
        )
    elif hasattr(raw_original_obj, "delegate_names") or hasattr(
        raw_original_obj, "check"
    ):
        # sklearn delegated method
        safe_patch(
            flavour_name, raw_original_obj, "fn", patched_fn,
        )
    else:
        # unsupported method type. skip patching
        pass
