import sys

import numpy as np

from sklearn.metrics import accuracy_score

from DataLoader import DataLoader
from models.CombinedModelsClassifier import CombinedModelsClassifier
from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier
from Utils import cofidence_interval, friedman_test

np.random.seed(42)

def main(argv):
	# Load and prepare the dataset.
	loader = DataLoader()
	loader.load(train_dir='database/segmentation.data.txt',  test_dir='database/segmentation.test.txt')

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
	print("")

	############

	print ("==  Friedman Test ==")
	friedman_test([bc, knn_classifier, combined_classfier],
				  X_train, y_train,
				  fit_params=[None,
				  	{'n_neighbors':1},
				  	{'n_neighbors':3, 'view1_columns' : view1_columns, 'view2_columns' : view2_columns}])


if __name__ == "__main__":
    main(sys.argv)