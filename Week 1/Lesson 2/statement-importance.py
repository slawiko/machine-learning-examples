import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier

default_data = pandas.read_csv('../titanic.csv', index_col='PassengerId')

data = pandas.DataFrame(data=default_data, columns=['Survived', 'Pclass', 'Fare', 'Age', 'Sex'])
data = data.dropna()
data = data.replace(to_replace='male', value=1)
data = data.replace(to_replace='female', value=0)

target_data = numpy.array(pandas.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex']))
target_variable = numpy.array(pandas.DataFrame(data=data, columns=['Survived']))

clf = DecisionTreeClassifier()
clf = clf.fit(target_data, target_variable)

print clf.feature_importances_
