import numpy
from sklearn import cross_validation as cv
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.preprocessing import scale

data = datasets.load_boston()
normalize_attributes = scale(data.data)

space = numpy.linspace(start=1, stop=10, num=200)

kfold = cv.KFold(n=normalize_attributes.__len__(), n_folds=5, random_state=42, shuffle=True)
max_quality = -50
minkowski_param = 0

for k in space:
	knr = KNR(n_neighbors=5, weights='distance', metric='minkowski', p=k)
	mark = cv.cross_val_score(knr, normalize_attributes, y=data.target, cv=kfold, scoring='mean_squared_error').mean()
	if mark > max_quality:
		max_quality = mark
		minkowski_param = k

print 'Optimal param: {}\nMax quality: {}'.format(minkowski_param, max_quality)
