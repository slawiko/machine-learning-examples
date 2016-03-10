import math

import numpy as np
import pandas
from sklearn.metrics import roc_auc_score


def grad_desc(k, X, y, xsize, C=0):
	w1, w2 = 0, 0
	i = 0
	e = 1
	while e > 0.00001 and i < 10000:
		summ1, summ2 = 0, 0
		w1_prev, w2_prev = w1, w2
		for j in range(xsize):
			summ1 += y[j] * X[j][0] * (1 - 1 / (1 + math.exp(-y[j] * (w1 * X[j][0] + w2 * X[j][1]))))
			summ2 += y[j] * X[j][1] * (1 - 1 / (1 + math.exp(-y[j] * (w1 * X[j][0] + w2 * X[j][1]))))
		w1 = w1 + k / xsize * summ1 - k * C * w1
		w2 = w2 + k / xsize * summ2 - k * C * w2
		i += 1
		e = math.sqrt((w1 - w1_prev) ** 2 + (w2 - w2_prev) ** 2)
	a = np.empty(xsize)
	for k in range(xsize):
		a[k] = 1 / (1 + math.exp(-w1 * X[k][0] - w2 * X[k][1]))
	return roc_auc_score(y_true=y, y_score=a)


data = pandas.read_csv(filepath_or_buffer='../data-logistic.csv', header=None)

grad = grad_desc(X=np.array(data.ix[:, 1:]), y=np.array(data[0]), k=0.1, xsize=data.__len__())
grad_reg = grad_desc(X=np.array(data.ix[:, 1:]), y=np.array(data[0]), k=0.1, xsize=data.__len__(), C=10)

print round(grad, 3), round(grad_reg, 3)
