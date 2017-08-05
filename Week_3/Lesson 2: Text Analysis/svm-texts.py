import numpy as np
from sklearn import cross_validation as cv
from sklearn import datasets
from sklearn import grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

tfidf = TfidfVectorizer()
newsgroups_data_tfidf = tfidf.fit_transform(newsgroups.data)

grid = {'C': np.power(10.0, np.arange(-5.0, 6.0))}
cross_validation = cv.KFold(n=newsgroups_data_tfidf.shape[0], n_folds=5, shuffle=True, random_state=241)
gs = grid_search.GridSearchCV(estimator=SVC(kernel='linear', random_state=241), param_grid=grid, scoring='accuracy',
                              cv=cross_validation)
gs.fit(X=newsgroups_data_tfidf, y=newsgroups.target)

max = 0
C = 0
for a in gs.grid_scores_:
	if a.mean_validation_score > max:
		max = a.mean_validation_score
		C = a.parameters['C']

clf = SVC(random_state=241, C=C, kernel='linear')
clf.fit(X=newsgroups_data_tfidf, y=newsgroups.target)

top_10_weights = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
top_10_words = np.sort(np.array(tfidf.get_feature_names())[top_10_weights])

print ','.join(top_10_words)
