import math
import sys

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from DataLoader import DataLoader
from models.CombinedModelsClassifier import CombinedModelsClassifier
from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier

N_TIMES = 30

np.random.seed(42)

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

def main(argv):
	# Load and prepare the dataset.
	loader = DataLoader()
	loader.load(train_dir='database/segmentation.data.txt', 
				test_dir='database/segmentation.test.txt')

	X_train, y_train, X_test, y_test = loader.data()

	# Confidence Interval for BayesianClassifier.
	bc = BayesianClassifier()
	min_interval, max_interval = cofidence_interval(bc, X_train, y_train)

	# Pontual estimative.
	bc.fit(X_train, y_train)
	y_pred = bc.predict(X_test)	

	print ("== Bayesian Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (min_interval,  max_interval))
	print ("")

	# Confidence interval for KNeighborsClassifier
	knn_classifier = KNNClassifier(n_neighbors=1)
	min_interval, max_interval = cofidence_interval(knn_classifier, X_train, y_train, fit_params={'n_neighbors':1})

	# Pontual estimative.
	knn_classifier.fit(X_train, y_train)
	y_pred = knn_classifier.predict(X_test)	

	print ("==  KNeighbors Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (min_interval,  max_interval))
	print ("")

	# Confidence interval for CombinedModelsClassifier
	view1_columns=[0,1,2,3,4,5,6,7,8]
	view2_columns=[9,10,11,12,13,14,15,16,17,18]
	combined_classfier = CombinedModelsClassifier(n_neighbors=3, view1_columns=view1_columns, view2_columns=view2_columns)
	min_interval, max_interval = cofidence_interval(combined_classfier, X_train, y_train,
													fit_params = {'n_neighbors':3, 'view1_columns' : view1_columns, 'view2_columns' : view2_columns})	

	# Pontual estimative.
	combined_classfier.fit(X_train, y_train)
	y_pred = combined_classfier.predict(X_test)

	print ("==  Combined Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (min_interval,  max_interval))

if __name__ == "__main__":
    main(sys.argv)