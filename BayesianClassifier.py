import sys

import numpy as np
from sklearn.naive_bayes import GaussianNB

class BayesianClassifier:
	""" Gaussian Bayesian classifier """
	
	def __init__(self, arg=None):
		self.arg = arg

	def fit(self, X, y):
		"""
		Fit a Gaussian Bayesian classifier based on X and y.

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

		assert(X.shape[0] == y.shape[0])

		classes = np.unique(y)
		n_classes = len(classes)
		n_features = X.shape[1]

		self.theta_ = np.zeros((n_classes, n_features))
		self.sigma_ = np.zeros((n_classes, n_features))
		self.class_count_ = np.zeros_like(classes)

		for c in classes:
			X_c = X[y == c, :] 
			N_c = X_c.shape[0]

			mu_c = np.mean(X_c, axis=0)
			self.theta_[c, :] = mu_c
			self.sigma_[c, : ] = np.mean(np.square(X_c - mu_c), axis=0)
			self.class_count_[c] = N_c
		
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

		assert(X.shape[1] == self.theta_.shape[1])

		C = np.zeros((X.shape[0], ), dtype=np.uint)
		P_w = self.class_count_ / np.sum(self.class_count_)

		sigma_inv = 1/self.sigma_

		for i, x_k in enumerate(X):
			C[i] = np.argmax(np.exp((-1/2) * np.sum((x_k - self.theta_) * sigma_inv * (x_k - self.theta_), axis=1)) * P_w)

		return C


def main(argv):
	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	Y = np.array([0, 0, 0, 1, 1, 1])

	bc = BayesianClassifier()
	bc.fit(X, Y)	

	clf = GaussianNB()
	clf.fit(X, Y)


	print(bc.predict(np.array([[-0.8, -1]])))
	print(clf.predict(np.array([[-0.8, -1]])))

if __name__ == "__main__":
    main(sys.argv)