import pandas
from sklearn import metrics

data = pandas.read_csv(filepath_or_buffer='../classification.csv')

TP, FP, FN, TN = 0, 0, 0, 0

eq = data[data['true'] == data['pred']]

TP = eq[eq['true'] == 1].shape[0]
TN = eq[eq['true'] == 0].shape[0]

not_eq = data[data['true'] != data['pred']]

FP = not_eq[not_eq['true'] == 0].shape[0]
FN = not_eq[not_eq['true'] == 1].shape[0]

print '1 task: {} {} {} {}'.format(TP, FP, FN, TN)

acc = metrics.accuracy_score(y_true=data['true'], y_pred=data['pred'])
prec = metrics.precision_score(y_true=data['true'], y_pred=data['pred'])
rec = metrics.recall_score(y_true=data['true'], y_pred=data['pred'])
f = metrics.f1_score(y_true=data['true'], y_pred=data['pred'])

print '2 task: {} {} {} {}'.format(round(acc, 2), round(prec, 2), round(rec, 2), round(f, 2))

data = pandas.read_csv(filepath_or_buffer='../scores.csv')

roc_logreg = metrics.roc_auc_score(y_true=data['true'], y_score=data['score_logreg'])
roc_svm = metrics.roc_auc_score(y_true=data['true'], y_score=data['score_svm'])
roc_knn = metrics.roc_auc_score(y_true=data['true'], y_score=data['score_knn'])
roc_tree = metrics.roc_auc_score(y_true=data['true'], y_score=data['score_tree'])

print '3 task: score_logreg'
print '4 task: score_tree'
