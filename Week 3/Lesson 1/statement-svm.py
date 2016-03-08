import pandas
from sklearn.svm import SVC

data = pandas.read_csv('../svm-data.csv', header=None)
target = data[0]
attributes = data.ix[:, 1:]

clf = SVC(random_state=241, C=100000, kernel='linear')
clf.fit(X=attributes, y=target)

print clf.support_
