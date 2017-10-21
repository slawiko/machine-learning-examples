# coding=utf-8
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation as cv
from sklearn import metrics
import pandas

# считываем данные

features = pandas.read_csv('features.csv', index_col='match_id')

features_test = pandas.read_csv('features_test.csv', index_col='match_id')

# ----------------------------------
start_time = datetime.datetime.now()
# ----------------------------------

# удаляем признаки, связанные с итогами матча (а также явно не необходимые признаки)

del features['duration']
del features['start_time']
del features['tower_status_radiant']
del features['tower_status_dire']
del features['barracks_status_radiant']
del features['barracks_status_dire']

del features_test['start_time']

# занулим пропуски

features = features.fillna(0)

features_test = features_test.fillna(0)

# выделим целевую переменную и матрицу объект-признак

y_train = features['radiant_win']
del features['radiant_win']
X_train = features
X_test = features_test

# обучим классификатор

clf = GradientBoostingClassifier(n_estimators=30, verbose=True, random_state=241, max_depth=2)
clf.fit(X=X_train, y=y_train)

# построим прдесказание

y_predict = clf.predict(X=X_test)

# проведем кросс-валидацию

kfold = cv.KFold(n=X_train.shape[0], n_folds=5, random_state=42, shuffle=True)
print cv.cross_val_score(estimator=clf, X=X_train, y=y_train, cv=kfold)

# print "AUC: ".format(metrics.roc_auc_score(y_true=y_train, y_score=y_predict))
# ---------------------------------------------------------
print 'Time elapsed:', datetime.datetime.now() - start_time
# ---------------------------------------------------------
