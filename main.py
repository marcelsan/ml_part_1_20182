import math
import sys

import numpy as np

from DataLoader import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier

N_TIMES = 30

np.random.seed(42)

def main(argv):
	# Load and prepare the dataset.
	loader = DataLoader()
	loader.load(train_dir='database/segmentation.data.txt', 
				test_dir='database/segmentation.test.txt')

	X_train, y_train, X_test, y_test = loader.data()

	# Confidence Interval for BayesianClassifier.
	bc = BayesianClassifier()
	ntimes_folds = np.zeros(N_TIMES)

	for i in range(N_TIMES):
		X, y = shuffle(X_train, y_train, random_state=i)
		skf = StratifiedKFold(n_splits=10)
		ntimes_folds[i] = np.mean(cross_val_score(bc, X, y, scoring='accuracy', cv=skf))

	f = np.mean(ntimes_folds)
	interval = 1.96 * math.sqrt( (f * (1 - f)) / N_TIMES)  # 95% confidance level.

	bc.fit(X_train, y_train)
	y_pred = bc.predict(X_test)	

	print ("== Bayesian Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % ( np.max((0.0, f - interval)), np.min((f + interval, 1.0)) ))
	print ("")

	# Confidence interval for KNeighborsClassifier
	knn_classifier = KNNClassifier(n_neighbors=1)
	ntimes_folds = np.zeros(N_TIMES)

	for i in range(N_TIMES):
		X, y = shuffle(X_train, y_train, random_state=i)
		skf = StratifiedKFold(n_splits=10)
		ntimes_folds[i] = np.mean(cross_val_score(knn_classifier, X, y, scoring='accuracy', cv=skf))

	f = np.mean(ntimes_folds)
	interval = 1.96 * math.sqrt( (f * (1 - f)) / N_TIMES)  # 95% confidance level.

	knn_classifier.fit(X_train, y_train)
	y_pred = knn_classifier.predict(X_test)	

	print ("==  KNeighbors Classifier ==")
	print("Estimativa pontual: %.3f" % accuracy_score(y_test, y_pred))
	print ("Intervalo de confiança: [%.3f, %.3f]" % ( np.max((0.0, f - interval)), np.min((f + interval, 1.0)) ))

if __name__ == "__main__":
    main(sys.argv)