import wandb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from wandb_sklearn_integration import autolog, eval_and_log_metrics

autolog()

# train a model
X, y = datasets.make_regression(n_features=10, n_informative=5, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with wandb.init(project="autolog", name="example_regression") as run:
    clf = LinearRegression()
    clf.fit(train_X, train_y) # this autologs train metrics
    eval_and_log_metrics(clf, test_X, test_y) # this autologs validation metrics
