from sklearn.feature_extraction import DictVectorizer
import pandas
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

train = pandas.read_csv('salary-train.csv')
test = pandas.read_csv('salary-test-mini.csv')

train['FullDescription'] = train['FullDescription'].str.lower()
test['FullDescription'] = test['FullDescription'].str.lower()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

td_if = TfidfVectorizer(min_df=5)

X = td_if.fit_transform(train['FullDescription'])

X_test = td_if.transform(test['FullDescription'])

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()

X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

Y = hstack([X, X_train_categ])
Y_test = hstack([X_test, X_test_categ])

clf = Ridge(alpha=1)
clf.fit(Y, train['SalaryNormalized'])
