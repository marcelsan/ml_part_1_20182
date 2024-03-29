import argparse
import sys

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from DataLoader import DataLoader
from models.CombinedModelsClassifier import CombinedModelsClassifier
from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier
from models.KCM_F_GH import KCM_F_GHClustering
from Utils import confidence_interval, friedman_test

def evaluate_classifiers(X_train, y_train, X_test, y_test):
	''' Evaluate the classification algorithms. '''

	# Initializes the views.
	VIEW1_COLUMNS = [0,1,2,3,4,5,6,7,8]
	VIEW2_COLUMNS = [9,10,11,12,13,14,15,16,17,18]

	print("+--------------------------------------+")
	print("|   Machine Learning Project, Part 2   |")
	print("+--------------------------------------+")

	############ BayesianClassifier  ############

	# Confidence Interval for BayesianClassifier.
	bc = BayesianClassifier()
	min_interval, max_interval, mean_accuracy = confidence_interval(bc, X_train, y_train)

	bc.fit(X_train, y_train)
	y_pred = bc.predict(X_test)	

	print ("=========== Bayesian Classifier ===========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print("30x 10-fold Accuracy Evaluation: %.3f" % mean_accuracy)
	print("Confidence Interval: [%.3f, %.3f]" % (min_interval,  max_interval))
	print ("")

	############ KNeighborsClassifier  ############

	# Confidence interval for KNeighborsClassifier
	knn_classifier = KNNClassifier(n_neighbors=1)
	min_interval, max_interval, mean_accuracy = confidence_interval(knn_classifier, X_train, y_train, fit_params={'n_neighbors':1})

	knn_classifier.fit(X_train, y_train)
	y_pred = knn_classifier.predict(X_test)	

	print ("==========  KNeighbors Classifier (k = 1) ==========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print("30x 10-fold Accuracy Evaluation: %.3f" % mean_accuracy)
	print("Confidence Interval: [%.3f, %.3f]" % (min_interval,  max_interval))
	print ("")

	############ CombinedModelsClassifier  ############

	# Confidence interval for CombinedModelsClassifier
	combined_classfier = CombinedModelsClassifier(n_neighbors=3, view1_columns=VIEW1_COLUMNS, view2_columns=VIEW2_COLUMNS)
	min_interval, max_interval, mean_accuracy = confidence_interval(combined_classfier, X_train, y_train,
													fit_params = {'n_neighbors':3, 'view1_columns' : VIEW1_COLUMNS, 'view2_columns' : VIEW2_COLUMNS})	

	combined_classfier.fit(X_train, y_train)
	y_pred = combined_classfier.predict(X_test)

	print ("===========  Combined Classifier (k = 3) ===========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print("30x 10-fold Accuracy Evaluation: %.3f" % mean_accuracy)
	print("Confidence Interval: [%.3f, %.3f]" % (min_interval,  max_interval))
	print("")

	############ Friedman Test  #################
	print ("==============  Friedman Test ==============")
	friedman_test([bc, knn_classifier, combined_classfier],
				  X_train, y_train, fit_params=[None, {'n_neighbors':1},
				  					{'n_neighbors':3, 'view1_columns' : VIEW1_COLUMNS, 'view2_columns' : VIEW2_COLUMNS}])

	print("")

	################################ DATASET VIEWS #######################################

	# We evaluate the classifers on Shape Veiw and RGB View separately.

	X_train_shape_view = X_train[:, VIEW1_COLUMNS]
	X_train_rgb_view = X_train[:, VIEW2_COLUMNS]

	X_test_shape_view = X_test[:, VIEW1_COLUMNS]
	X_test_rgb_view = X_test[:, VIEW2_COLUMNS]


	print("======== Shape View ========")
	
	############ BayesianClassifier  ############
	bc = BayesianClassifier()
	y_pred = bc.fit(X_train_shape_view, y_train).predict(X_test_shape_view)	
	
	print ("=========== Bayesian Classifier ===========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print ("")

	############ KNeighborsClassifier  ############
	knn_classifier = KNNClassifier(n_neighbors=1)
	y_pred = knn_classifier.fit(X_train_shape_view, y_train).predict(X_test_shape_view)

	print ("==========  KNeighbors Classifier (k = 1) ==========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print ("")




	print("========= RGB View =========")

	############ BayesianClassifier  ############
	bc = BayesianClassifier()
	y_pred = bc.fit(X_train_rgb_view, y_train).predict(X_test_rgb_view)	

	print ("=========== Bayesian Classifier ===========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print ("")

	############ KNeighborsClassifier  ############
	knn_classifier = KNNClassifier(n_neighbors=1)
	y_pred = knn_classifier.fit(X_train_rgb_view, y_train).predict(X_test_rgb_view)	

	print ("==========  KNeighbors Classifier (k = 1) ==========")
	print("Accuracy Evaluation on Test Set : %.3f" % accuracy_score(y_test, y_pred))
	print ("")

def evaluate_clustering(X, y, n_executions = 100):
	''' Evaluate the clustering algorithms. '''

	# Initializes the views.
	# We removed the column 2 because the KCM method is not suitable to features with
	# all repeated values.
	
	SHAPE_VIEW_COLUMNS = [0,1,3,5,6,7,8]
	RGB_VIEW_COLUMNS = [9,10,11,12,13,14,15,16,17,18]
	FULL_VIEW_COLUMNS = SHAPE_VIEW_COLUMNS + RGB_VIEW_COLUMNS

	X_shape_view = X[:, SHAPE_VIEW_COLUMNS]
	X_rgb_view = X[:, RGB_VIEW_COLUMNS]
	X_full_view = X[:, FULL_VIEW_COLUMNS]

	print("+--------------------------------------+")
	print("|   Machine Learning Project, Part 1   |")
	print("+--------------------------------------+")

	############ Shape View  ############
	print ("== Shape View ==")
	kcm = max(
		map(lambda x: KCM_F_GHClustering(c=7).fit(X_shape_view), range(n_executions)),
		key = lambda kcm: adjusted_rand_score(kcm.predict(X_shape_view), y))
	report(kcm, X_shape_view, y)

	############ RGB View  ############
	print ("== RGB View ==")
	kcm = max(
		map(lambda x: KCM_F_GHClustering(c=7).fit(X_rgb_view), range(n_executions)),
		key = lambda kcm: adjusted_rand_score(kcm.predict(X_rgb_view), y))
	report(kcm, X_rgb_view, y)

	############ Full View  ############
	print ("== Full View ==")
	kcm = max(
		map(lambda x: KCM_F_GHClustering(c=7).fit(X_full_view), range(n_executions)),
		key = lambda kcm: adjusted_rand_score(kcm.predict(X_full_view), y))
	report(kcm, X_full_view, y)


def report(kcm, X, y):
	np.set_printoptions(precision = 4, linewidth = 200)

	print("Number of elements per cluster:")
	for i, cluster in enumerate(kcm.clusters):
		print("cluster " + str(i) + " has " + str(len(cluster)) + " elements")
	
	print("Hyperparameter vector (1 / s^2):")
	print(kcm.inv_s2_)

	print("Element indices per cluster:")
	for i, cluster in enumerate(kcm.clusters):
		print("cluster " + str(i) + ":")
		print(cluster)

	print("Elements per cluster:")
	for i, cluster in enumerate(kcm.clusters):
		print("cluster " + str(i) + ":")
		print(X[cluster])

	kcm_labels = kcm.predict(X)
	score = adjusted_rand_score(kcm_labels, y)
	print("Rand Score: %.3f" %score)
	
	np.set_printoptions()

def main(args):
	#np.random.seed(100)
	# Load and prepare the dataset.
	loader = DataLoader()
	loader.load(train_dir='database/segmentation.data.txt',  test_dir='database/segmentation.test.txt')

	X_train, y_train, X_test, y_test = loader.data()

	if not args.eval_classifiers and not args.eval_clustering:
		evaluate_classifiers(X_train, y_train, X_test, y_test)
		evaluate_clustering(X_test, y_test)
	else:
		if args.eval_classifiers:
			evaluate_classifiers(X_train, y_train, X_test, y_test)

		if args.eval_clustering:
			evaluate_clustering(X_test, y_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--eval-classifiers', help='Evaluate the classification algorithms.', action='store_true', default=False)
	parser.add_argument('--eval-clustering', help='Evaluate the clustering algorithms.', action='store_true', default=False)
	args = parser.parse_args()

	main(args)