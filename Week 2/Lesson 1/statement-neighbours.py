import pandas
from sklearn import cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.preprocessing import scale

data = pandas.read_csv(filepath_or_buffer='../wine.data', header=None)

attributes = data.ix[:, 1:]
classes = data[0]
kfold = cv.KFold(n=178, n_folds=5, random_state=42, shuffle=True)

attributes = scale(attributes)
max = 0
k = 1
k_neighbor = None
while k <= 50:
	knc = KNC(n_neighbors=k)
	m = cv.cross_val_score(knc, attributes, classes, cv=kfold).mean()
	print '{} neighbor: {}'.format(k, m)
	if m > max:
		max = m
		k_neighbor = k
	k += 1

print 'Maximum: {}\nNeighbors: {}'.format(max, k_neighbor)
