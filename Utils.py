import math

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def cofidence_interval(clf, X, y, n_times=30, k=10, fit_params=None):
	"""
	Obtain the accuracy confidence interval of a given classifier.
	
	Parameters
	---------
	clf : 		The classifier that we want obtain the accuracy cofidence interval.

	X : 		shape (n_samples, n_features)
				Data vectors, where n_samples is the number of samples
				and  n_features is the number of features.

	y : 		shape (n_samples, )
				Target values factorized. Each class must have a 0-index value.

	n_times : 	Int (default=30)
				The number of k-fold that are evaluated. Default is 30 times.

	k :			Int
				Determines the cross-validation splitting strategy. 
				Default is 10-fold cross validation.

	fit_params: dict (optional)
				Parameters to pass to the fit method of the estimator.

	Returns
	---------
	self :	 	pair
				The confidence interval.

	"""
	
	ntimes_folds = np.zeros(n_times)

	for i in range(n_times):
		X_, y_ = shuffle(X, y, random_state=i)
		skf = StratifiedKFold(n_splits=k)
		ntimes_folds[i] = np.mean(cross_val_score(clf, X_, y_, scoring='accuracy', cv=skf, fit_params=fit_params))

	f = np.mean(ntimes_folds)
	interval = 1.96 * math.sqrt( (f * (1 - f)) / n_times)  # 95% confidance level.

	return (np.max((0.0, f - interval)), np.min((f + interval, 1.0)))

def friedman_test(clfs, X, y, fit_params=None):
	N = 30 
	k = len(clfs)
	df = k-1 # Degrees of freedom

	assert(k == 3) # In this case, we know beforehand that we compare 3 classifiers.

	ntimes_folds = np.zeros((N, k))
	for i, clf in enumerate(clfs):
		for j in range(N):
			X_, y_ = shuffle(X, y, random_state=j)
			skf = StratifiedKFold(n_splits=10)
			ntimes_folds[j, i] = np.mean(cross_val_score(clf, X_, y_, scoring='accuracy', cv=skf, fit_params=fit_params[i]))

	ntimes_folds = np.argsort(ntimes_folds) + 1
	ranks = np.sum(ntimes_folds, axis=0)/N
	ranks_ = ranks - (k+1)/2

	X_r2 = (12*N/(k*(k+1))) * np.sum(ranks_ ** 2)

	if X_r2 > 5.991: # (Qui-squared k=2, for alpha = 0.95 )
		print("Reject. There is a difference between the three classifiers.")

		CD =  2.344 * np.sqrt((k*(k+1))/(6 * N))

		# Compare classifier 0 to 1
		if np.abs(ranks[0] - ranks[1]) >= CD:
			print("The classifier %s is different to %s." %(clfs[0].name_, clfs[1].name_))

		# Compare classifier 1 to 2
		if np.abs(ranks[1] - ranks[2]) >= CD:
			print("The classifier %s is different to %s." %(clfs[1].name_, clfs[2].name_))

		# Compare classifier 0 to 2
		if np.abs(ranks[0] - ranks[2]) >= CD:
			print("The classifier %s is different to %s." %(clfs[0].name_, clfs[2].name_))

	else:
		print("Do not reject.")

