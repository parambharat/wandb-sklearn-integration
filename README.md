
# wandb-sklearn-integration

An auto logging integration for [scikit-learn](https://scikit-learn.org/stable/), that logs metrics, models and plots into [Weights & Biases](https://wandb.ai/).


## Features
The integration currently has the following features:

- Log train and eval metrics for all sklearn estimators
- Logs model artifacts, parameters and configurations
- Logs common classification plots (e.g. pr-curve, confusion-matrix)
- Logs GridSearchCV and other meta estimator results into a wandb.Table


## Installation

Install with pip from github.

```bash
  pip install git+https://github.com/parambharat/wandb-sklearn-integration#egg=wandb_sklearn_integration
```

## Usage/Examples

```python
import wandb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wandb_sklearn_integration import autolog, eval_and_log_metrics

# Call the autolog function at the start of your script or before training any models
autolog()

# load your classification dataset
X, y = datasets.make_classification(n_features=10, n_informative=5, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# initialize a wandb run
with wandb.init(project="autolog") as run:
    clf = LogisticRegression()
    clf.fit(train_X, train_y) # this autologs train metrics, model artifacts and plots
    eval_and_log_metrics(clf, test_X, test_y) # this autologs validation metrics and plots

```
Checkout the metrics logged on your [Weights & Biases dashboard](https://wandb.ai/parambharat/autolog?workspace=user-parambharat)

For more usage examples checkout the [examples](https://github.com/parambharat/wandb-sklearn-integration/tree/main/examples) directory

## Inspiration
This autologger is largely inspired from [MLflow's](https://mlflow.org/) autologger for scikit-learn.
