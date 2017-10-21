import pandas
import math
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

data = pandas.read_csv('gbm-data.csv')

train = data.drop('Activity', 1)
target = data['Activity']

train = train.values
target = target.values

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.8, random_state=241)

list = [1, 0.5, 0.3, 0.2, 0.1]

for i in list:
	clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
	for predict in clf.staged_decision_function(X=X_train):
		predict = 1 + math.exp(- predict)

		test_loss = log_loss(y_true=y_train, y_pred=predict)

