import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

train = data.drop('Rings', 1)
target = data['Rings']

for i in range(1, 51):
	cv = KFold(n=train.shape[0], n_folds=5, shuffle=True, random_state=1)
	rfregressor = RandomForestRegressor(random_state=1, n_estimators=i)
	rfregressor.fit(X=train, y=target)
	metric = cross_val_score(estimator=rfregressor, scoring='r2', X=train, y=target, cv=cv).mean()
	if metric > 0.52:
		print 'Tree number: {}; metric score: {}'.format(i, metric)
		break
