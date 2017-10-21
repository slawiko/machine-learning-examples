# coding=utf-8
"""
Created on Sat Mar 12 15:59:14 2016

@author: Gorokhov
"""

import numpy as np
import pandas as pn
from sklearn import cross_validation
from sklearn import ensemble, linear_model, grid_search
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

features = pn.read_csv('../../features.csv', index_col='match_id')

# Вычисляем длину массива

feature_len = features.shape[0]

p = features.count().to_frame()
p.columns = ['value']
names = list(p.columns.values)
k1 = p.loc[(p['value'] < feature_len)]
k1['value'] = feature_len - k1['value']

features = features.fillna(0)
X = features.copy()

# Удаляем лишние фичи
X = X.drop({'radiant_win', 'duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant',
            'barracks_status_dire'}, axis=1)
y = np.asarray(features[['radiant_win']]).ravel()
l = len(y)
# Градиентный бустинг
import datetime

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
# sub=features.select(lambda x:  'time' in x, axis=1)

kf = KFold(l, n_folds=5, shuffle=True, random_state=51)
lr = np.array([10, 20, 30])
itog_val = {}
itog_roc = {}

# Проведем кроссвалидацию на разном колличестве деревьев.

for j in range(len(lr)):
	start_time = datetime.datetime.now()
	clf = ensemble.GradientBoostingClassifier(n_estimators=lr[j], verbose=True, random_state=51)
	scores = cross_validation.cross_val_score(clf, X, y, cv=kf)
	itog_val[str(lr[j])] = scores.mean()
	print('Время настройки кросс-валидации:', datetime.datetime.now() - start_time)
	print('Колличество деревьев: ', lr[j])
	start_time = datetime.datetime.now()
	clf.fit(X_train, y_train)
	print('Время тренировки:', datetime.datetime.now() - start_time)
	probas = clf.predict_proba(X_test)
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
	roc_auc = metrics.auc(fpr, tpr)
	itog_roc[str(lr[j])] = roc_auc

# Логистическая регрессия

grid = {'C': np.power(10.0, np.arange(-5, 6))}
logreg = linear_model.LogisticRegression(penalty='l2')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

itog_c = {}

gs = grid_search.GridSearchCV(logreg, grid, scoring='accuracy', cv=kf)
gs.fit(X_scaled, y)
print('Регрессия на неизмененном наборе данных')
for a in gs.grid_scores_:
	print('оценка качества по кросс-валидации ', a.mean_validation_score)
	print('значение параметра С ', a.parameters)
pr = gs.predict_proba(X_scaled)
fpr, tpr, thresholds = metrics.roc_curve(y, pr[:, 1])
reg_roc_auc1 = metrics.auc(fpr, tpr)
print('ROC AUC', reg_roc_auc1)
bestC = gs.best_params_
itog_c['first'] = bestC

print('Лучшее значение параметра', bestC)

# Удаляем героевй
subX = X.select(lambda x: 'hero' not in x, axis=1)
subX_scaled = scaler.fit_transform(subX)

gs.fit(subX_scaled, y)
print('Регрессия "без героев"')
for a in gs.grid_scores_:
	print('оценка качества по кросс-валидации ', a.mean_validation_score)
	print('значение параметра С ', a.parameters)
pr = gs.predict_proba(subX_scaled)
fpr, tpr, thresholds = metrics.roc_curve(y, pr[:, 1])
reg_roc_auc2 = metrics.auc(fpr, tpr)
print('ROC AUC', reg_roc_auc2)

bestC = gs.best_params_
itog_c['second'] = bestC
print('Лучшее значение параметра', bestC)
heroX = X.select(lambda x: 'hero' in x, axis=1)

ch = [s for s in X.columns if "hero" in s]
hu = np.unique(X[ch])


def makeBag(data, N):
	bag = np.zeros((data.shape[0], N))

	for i, match_id in enumerate(data.index):
		for p in range(5):
			bag[i, data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
			bag[i, data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

	return bag


heroCount = np.max(pn.unique(heroX.as_matrix().ravel()))
bag = makeBag(heroX, heroCount)
wordsBag = np.hstack((subX_scaled, bag))
gs.fit(wordsBag, y)
print('Регрессия с мешком слов')
for a in gs.grid_scores_:
	print('оценка качества по кросс-валидации ', a.mean_validation_score)
	print('значение параметра С ', a.parameters)
pr = gs.predict_proba(wordsBag)
fpr, tpr, thresholds = metrics.roc_curve(y, pr[:, 1])
reg_roc_auc3 = metrics.auc(fpr, tpr)
print('ROC AUC', reg_roc_auc3)
bestC = gs.best_params_
print('Лучшее значение параметра', bestC)
itog_c['third'] = bestC
print('Время :', datetime.datetime.now() - start_time)
reg = linear_model.LogisticRegression(penalty='l2', C=bestC['C'])
X_train, X_test, y_train, y_test = train_test_split(wordsBag, y, test_size=0.3, random_state=51)
start_time = datetime.datetime.now()
reg.fit(X_train, y_train)
print('Время тренировки:', datetime.datetime.now() - start_time)
probas = reg.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
reg_roc_auc = metrics.auc(fpr, tpr)

minv = min(pr[:, 1])
print('Минимальное значение:', minv)
maxv = max(pr[:, 1])
print('Максимальное значение:', maxv)
