import sys

import numpy as np

from sklearn.metrics import accuracy_score, adjusted_rand_score
from DataLoader import DataLoader
from models.CombinedModelsClassifier import CombinedModelsClassifier
from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier
from models.KCM_F_GH import KCM_F_GHClustering
from Utils import cofidence_interval, friedman_test

np.random.seed(42)

# Initializes the views.
VIEW1_COLUMNS = [0,1,2,3,4,5,6,7,8]
VIEW2_COLUMNS = [9,10,11,12,13,14,15,16,17,18]

def evaluate_classifiers(X_train, y_train, X_test, y_test):

	############ BayesianClassifier  ############

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

	############ KNeighborsClassifier  ############

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

	############ CombinedModelsClassifier  ############

	# Confidence interval for CombinedModelsClassifier
	combined_classfier = CombinedModelsClassifier(n_neighbors=3, view1_columns=VIEW1_COLUMNS, view2_columns=VIEW2_COLUMNS)
	min_interval, max_interval = cofidence_interval(combined_classfier, X_train, y_train,
													fit_params = {'n_neighbors':3, 'view1_columns' : VIEW1_COLUMNS, 'view2_columns' : VIEW2_COLUMNS})	

	# Pontual estimative.
	combined_classfier.fit(X_train, y_train)
	y_pred = combined_classfier.predict(X_test)

	print ("==  Combined Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (min_interval,  max_interval))
	print("")

	############ Friedman Test  ############

	print ("==  Friedman Test ==")
	friedman_test([bc, knn_classifier, combined_classfier],
				  X_train, y_train,
				  fit_params=[None,
				  	{'n_neighbors':1},
				  	{'n_neighbors':3, 'view1_columns' : VIEW1_COLUMNS, 'view2_columns' : VIEW2_COLUMNS}])

def main(argv):
	# Load and prepare the dataset.
	loader = DataLoader()
	loader.load(train_dir='database/segmentation.data.txt',  test_dir='database/segmentation.test.txt')

	X_train, y_train, X_test, y_test = loader.data()

	evaluate_classifiers(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main(sys.argv)