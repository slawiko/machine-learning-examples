### PLASE NOTE THAT SOME CODE IS COMMENTED OUT FOR SIMPLICITY. IT CAN BE UNCOMMENTED PART-BY-PART AND CHECKED ###



import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

features = pandas.read_csv('./features.csv', index_col='match_id')

# calculate number of different heroes
hero_columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
heroes_unique = list(np.unique(features[hero_columns]))
hero_number = len(heroes_unique)
print "number of heroes = %s" % hero_number

y = features['radiant_win']
future_features = ['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant',
                   'barracks_status_dire']
features.drop(future_features, inplace=True, axis=1)  # drop features related to the outcome of the game

features_num = len(features)  # number of items in the train set
print features.count()[features.count() < features_num]  # printing features that have missing values

features = features.fillna(0)  # filling missing elements with 0 values
assert len(features.count()[features.count() < features_num]) == 0  # make sure all nan elements are filled

# GRADIENT BOOSTING
# num_estimators = [10, 20, 30]
# for n_est in num_estimators:
# start_time = datetime.datetime.now()
# cross_val = KFold(n_folds=5, n=len(features), shuffle=True)
#     classifier = GradientBoostingClassifier(n_estimators=n_est)
#     score_val = 0
#     for train, test in cross_val:
#         classifier.fit(features.iloc[train, :], y.iloc[train])
#         predicted = classifier.predict_proba(features.iloc[test, :])[:, 1]
#         score_val += roc_auc_score(y.iloc[test], predicted)
#     score_val /= cross_val.n_folds
#
#     print("n_estimators = %s, score = %s" % (n_est, score_val))
#     print 'Time elapsed:', datetime.datetime.now() - start_time


# LOGISTIC REGRESSION
# regularizers = [0.01, 0.1, 0.5, 1, 10, 100]
# for regularizer in regularizers:
#     start_time = datetime.datetime.now()
#     scaler = StandardScaler()
#     cross_val = KFold(n_folds=5, n=len(features), shuffle=True)
#     classifier = LogisticRegression(C=regularizer)
#     score_val = 0
#     for train, test in cross_val:
#         classifier.fit(scaler.fit_transform(features.iloc[train, :]), y.iloc[train])
#         predicted = classifier.predict_proba(scaler.transform(features.iloc[test, :]))[:, 1]
#         score_val += roc_auc_score(y.iloc[test], predicted)
#     score_val /= cross_val.n_folds
#
#     print "C = %s. Score = %s" % (regularizer, score_val)
#     print 'Time elapsed:', datetime.datetime.now() - start_time


# remove categorical features
# categorical_features = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
#                         'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
# features.drop(categorical_features, inplace=True, axis=1)

# LOGISTIC REGRESSION WITHOUT CATEGORICAL FEATURES
# regularizers = [0.01, 0.1, 0.5, 1, 10, 100]
# for regularizer in regularizers:
#     start_time = datetime.datetime.now()
#     scaler = StandardScaler()
#     cross_val = KFold(n_folds=5, n=len(features), shuffle=True)
#     classifier = LogisticRegression(C=regularizer)
#     score_val = 0
#     for train, test in cross_val:
#         classifier.fit(scaler.fit_transform(features.iloc[train, :]), y.iloc[train])
#         predicted = classifier.predict_proba(scaler.transform(features.iloc[test, :]))[:, 1]
#         score_val += roc_auc_score(y.iloc[test], predicted)
#     score_val /= cross_val.n_folds
#
#     print "C = %s. Score = %s" % (regularizer, score_val)
#     print 'Time elapsed:', datetime.datetime.now() - start_time


# convert categorical features
X_pick = np.zeros((features.shape[0], hero_number))
for i, match_id in enumerate(features.index):
	for p in xrange(5):
		X_pick[i, heroes_unique.index(features.ix[match_id, 'r%d_hero' % (p + 1)])] = 1
		X_pick[i, heroes_unique.index(features.ix[match_id, 'd%d_hero' % (p + 1)])] = -1

# LOGISTIC REGRESSION WITH CATEGORICAL FEATURES
# regularizers = [0.01, 0.1, 0.5, 1, 10, 100]
# for regularizer in regularizers:
#     start_time = datetime.datetime.now()
#     scaler = StandardScaler()
#     cross_val = KFold(n_folds=5, n=len(features), shuffle=True)
#     classifier = LogisticRegression(C=regularizer)
#     score_val = 0
#     for train, test in cross_val:
#         trX = np.concatenate((features.iloc[train, :], X_pick[train, :]), axis=1)
#         tstX = np.concatenate((features.iloc[test, :], X_pick[test, :]), axis=1)
#         classifier.fit(scaler.fit_transform(trX), y.iloc[train])
#         predicted = classifier.predict_proba(scaler.transform(tstX))[:, 1]
#         score_val += roc_auc_score(y.iloc[test], predicted)
#     score_val /= cross_val.n_folds
#
#     print "C = %s. Score = %s" % (regularizer, score_val)
#     print 'Time elapsed:', datetime.datetime.now() - start_time


# CONSTRUCT FINAL PREDICTOR


scaler = StandardScaler()
classifier = LogisticRegression(C=10)
trainX = np.concatenate((features.iloc[:, :], X_pick[:, :]), axis=1)
classifier.fit(scaler.fit_transform(trainX), y.iloc[:])

testX = pandas.read_csv('./features_test.csv', index_col='match_id')
testX = testX.fillna(0)
X_pick_test = np.zeros((testX.shape[0], hero_number))
for i, match_id in enumerate(testX.index):
	for p in xrange(5):
		X_pick_test[i, heroes_unique.index(testX.ix[match_id, 'r%d_hero' % (p + 1)])] = 1
		X_pick_test[i, heroes_unique.index(testX.ix[match_id, 'd%d_hero' % (p + 1)])] = -1

testX = np.concatenate((testX, X_pick_test), axis=1)
resProb = classifier.predict_proba(scaler.transform(testX))

print np.min(resProb)
print np.max(resProb)
