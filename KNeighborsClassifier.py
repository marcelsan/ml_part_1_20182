import sys

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
	"""
	K-Nearest Neighbors Classifier
	Classifier implementing the k-nearest neighbors vote.
	
	Parameters
	---------
	n_neighbors : int, optional (default = 3)
		Number of neighbors to use by default.

	"""

	def __init__(self, n_neighbors=3):
		self.n_neighbors_ = n_neighbors
	
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
			idx = np.argsort(dist)[0:self.n_neighbors_]
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

def main(argv):
	np.random.seed(0)

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

if __name__ == "__main__":
    main(sys.argv)