import sys

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

class KNNClassifier(BaseEstimator, ClassifierMixin):
	"""
	K-Nearest Neighbors Classifier
	Classifier implementing the k-nearest neighbors vote.
	
	Parameters
	---------
	n_neighbors : int, optional (default = 3)
		Number of neighbors to use by default.

	"""

	def __init__(self, n_neighbors=3):
		self.n_neighbors = n_neighbors
	
	def fit(self, X, y):
		"""
		Store in the class the training dataset.

		Parameters
		---------
		X : shape (n_samples, n_features)
			Training vectors, where n_samples is the number of training samples
			and  n_features is the number of features.
		y : shape (n_samples, )
			Target values factorized. Each class must have a 0-index value.

		Returns
		---------
		self : object
		"""

		self.X_ = X
		self.y_ = y

		return self

	def predict(self, X):
		"""
		Perform classification on an array of test vectors X.

		Parameters
		---------
		X : shape (n_samples, n_features)
			Array of test vectors.
	
		Returns
		---------
		C : shape (n_samples, )
			Predicted target values for array X. Class label for
			each data sample.

		"""

		assert(X.shape[1] == self.X_.shape[1])

		C = np.zeros((X.shape[0], ), dtype=np.uint)

		for i, x_k in enumerate(X):
			dist = np.sum(np.square(self.X_ - x_k), axis=1)
			idx = np.argsort(dist)[0:self.n_neighbors]
			C[i] = self.majority_vote_(self.y_[idx])


		return C

	def majority_vote_(self, list_vals):
		"""
		Perform majority vote on the input array.

		Parameters
		---------
		list_vals : Array of int.
	
		Returns
		---------
		a: int
			The element that appears most frequently on the array

		"""

		as_ = list_vals.tolist()
		a = max(map(lambda val: (as_.count(val), val), set(as_)))[1]

		return a

def print_scores(grid_search):
	cvres = grid_search.cv_results_
	for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	    print(mean_score, params)

def main(argv):
	np.random.seed(0)

	# Test the classifier
	knn = KNNClassifier(n_neighbors=3)
	iris = datasets.load_iris()
	iris_X = iris.data
	iris_y = iris.target

	indices = np.random.permutation(len(iris_X))
	iris_X_train = iris_X[indices[:-10]]
	iris_y_train = iris_y[indices[:-10]]
	iris_X_test  = iris_X[indices[-10:]]
	iris_y_test  = iris_y[indices[-10:]]

	knn.fit(iris_X_train, iris_y_train) 
	print(knn.predict(iris_X_test))

	# Use the GridSearch for choosing the best n_neighbors 
	# parameter on the Image Segmentation dataset.
	
	# Load data.
	image_segmentation = pd.read_csv('database/segmentation.data.txt', delimiter=',')
	image_segmentation = shuffle(image_segmentation)

	X_train = image_segmentation.drop("CLASS", axis=1)
	y_train, _ = image_segmentation["CLASS"].copy().factorize()

	shape_view_variables = ["REGION-CENTROID-COL", "REGION-CENTROID-ROW",
							"REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5",
							"SHORT-LINE-DENSITY-2", "VEDGE-MEAN","VEDGE-SD",
							"HEDGE-MEAN", "HEDGE-SD"]

	rgb_view_variables = ["INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", 
							"RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", 
							"EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN",
							"HUE-MEAN"]


	# Set the grid search.
	params = {
    	'n_neighbors': [1, 3, 5, 7, 9, 11, 13]
	}

	knn_classifier = KNNClassifier()

	grid_search = GridSearchCV(knn_classifier, param_grid=params, cv=5, scoring="accuracy")
	
	# Show the results for each view.

	print("[INFO] Complete View")
	grid_search.fit(X_train.values, y_train)
	print_scores(grid_search)
	print(grid_search.best_params_)
	print("--------------------------------------------")

	print("[INFO] RGB View")
	grid_search.fit(X_train[rgb_view_variables].values, y_train)
	print_scores(grid_search)
	print(grid_search.best_params_)
	print("--------------------------------------------")

	print("[INFO] Shape View")
	grid_search.fit(X_train[shape_view_variables].values, y_train)
	print_scores(grid_search)
	print(grid_search.best_params_)
	print("--------------------------------------------")

if __name__ == "__main__":
    main(sys.argv)