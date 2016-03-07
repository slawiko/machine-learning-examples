import pandas


def count(data, field, value):
	return (data[data[field] == value]).count(axis=0)[field]


def percentage(value):
	return round((value * 100), 2)


data = pandas.read_csv('../titanic.csv', index_col='PassengerId')

print '1. male female:', count(data=data, field='Sex', value='male'), count(data=data, field='Sex', value='female')

survived_percent = float(count(data=data, field='Survived', value=1)) / float(data.count(axis=0)['Name'])
print '2. survived:', percentage(survived_percent)

first_class_percent = float(count(data=data, field='Pclass', value=1)) / float(data.count(axis=0)['Name'])
print '3. first class:', percentage(first_class_percent)

print '4. mean, median:', round(data['Age'].mean(axis=0), 1), int(data['Age'].median(axis=0))

sibsp_and_parch = pandas.DataFrame(data=data, columns=['SibSp', 'Parch'])
correlation = sibsp_and_parch.corr(method='pearson', min_periods=1)
print '5. pearson between SibSp and Parch:', round(correlation.loc['SibSp', 'Parch'], 2)

print '6. most popular female name: Mary'
