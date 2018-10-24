import math
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier

N_TIMES = 30

np.random.seed(42)

def main(argv):
	image_segmentation = pd.read_csv('database/segmentation.data.txt', delimiter=',')
	image_segmentation = shuffle(image_segmentation)

	X_train = image_segmentation.drop("CLASS", axis=1).values
	y_train, _ = image_segmentation["CLASS"].copy().factorize()

	# Confidence Interval for BayesianClassifier.
	bc = BayesianClassifier()
	ntimes_folds = np.zeros(N_TIMES)

	for i in range(N_TIMES):
		X, y = shuffle(X_train, y_train, random_state=i)
		skf = StratifiedKFold(n_splits=10)
		ntimes_folds[i] = np.mean(cross_val_score(bc, X, y, scoring='accuracy', cv=skf))

	f = np.mean(ntimes_folds)
	interval = 1.96 * math.sqrt( (f * (1 - f)) / N_TIMES)  # 95% confidance level.

	print ("== Bayesian Classifier ==")
	print("Estimativa pontual: %.3f" % (f))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (f - interval, f + interval))
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

	print ("==  KNeighbors Classifier ==")
	print("Estimativa pontual: %.3f" % (f))
	print ("Intervalo de confiança: [%.3f, %.3f]" % (f - interval, f + interval))


if __name__ == "__main__":
    main(sys.argv)