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
with wandb.init(project="autolog", name="example_classification") as run:
    clf = LogisticRegression()
    clf.fit(train_X, train_y) # this autologs train metrics, model artifacts and plots
    eval_and_log_metrics(clf, test_X, test_y) # this autologs validation metrics and plots