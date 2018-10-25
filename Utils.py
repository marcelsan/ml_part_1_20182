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