import wandb

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from wandb_sklearn_integration import autolog, eval_and_log_metrics

autolog()


iris = datasets.load_iris()
train_X, test_X, train_y, test_y = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with wandb.init(project="autolog",) as run:
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(train_X, train_y)
    eval_and_log_metrics(clf, test_X, test_y)
