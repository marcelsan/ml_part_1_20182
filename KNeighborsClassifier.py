import sys

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
	"""K-Nearest Neighbors Classifier"""

	def __init__(self, n_neighbors=3):
		self.n_neighbors_ = n_neighbors
	
	def fit(self, X, y):
		self.X_ = X
		self.y_ = y

	def predict(self, X):

		assert(X.shape[1] == self.X_.shape[1])

		preds = np.zeros((X.shape[0], ))

		for i, x_k in enumerate(X):
			dist = np.sum(np.square(self.X_ - x_k), axis=1)
			idx = np.argsort(dist)[0:self.n_neighbors_]
			preds[i] = self.most_common_(self.y_[idx])

		return preds

	def most_common_(self, list_vals):
		a = list_vals.tolist()
		return max(map(lambda val: (a.count(val), val), set(a)))[1]

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