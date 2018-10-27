import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB

class BayesianClassifier(BaseEstimator, ClassifierMixin):
	""" Gaussian Bayesian classifier """
	
	def __init__(self):
		self.fitted = False
		self.name_ = "BayesianClassifier"

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
		self.n_features_ = X.shape[1]

		self.theta_ = np.zeros((n_classes, self.n_features_))
		self.sigma_ = np.zeros((n_classes, self.n_features_))
		self.class_count_ = np.zeros_like(classes)

		for c in classes:
			X_c = X[y == c] 
			N_c = X_c.shape[0]

			mu_c = np.mean(X_c, axis=0)

			self.theta_[c, :] = mu_c
			self.sigma_[c, : ] = np.mean(np.square(X_c - mu_c), axis=0)
			self.class_count_[c] = N_c
		
		self.pw_ = self.class_count_ / np.sum(self.class_count_)
		self.fitted = True		
		self.sigma_inv = 1/(self.sigma_ + 1e-5)

		# Pre-calculate the first multiplication term of the p(x|w) 
		det_sigma_abs = np.abs(np.prod(self.sigma_, axis=1))
		self.A = 1/(np.sqrt(np.power(2 * np.pi, self.n_features_) * (det_sigma_abs)) + 1e-5)

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

		if self.fitted is False:
			raise RuntimeError('The estimator was not fittet. Consider call .fit() first.')

		assert(X.shape[1] == self.theta_.shape[1])

		C = np.zeros((X.shape[0], ), dtype=np.uint)	

		for i, x_k in enumerate(X):
			likelihood = self.A * np.exp((-1/2) * np.sum((x_k - self.theta_) * self.sigma_inv * (x_k - self.theta_), axis=1))
			C[i] = np.argmax(likelihood * self.pw_)

		return C

	def predict_proba(self, X):
		"""
		Return probability estimates for the test data X.

		Parameters
		---------
		X : array of shape (n_samples, n_features)
			Array of test vectors.

		Returns
		---------
		p : array of shape (n_samples, n_classes)
			The class probabilities of the input sample.
			
		"""

		if self.fitted is False:
			raise RuntimeError('The estimator was not fittet. Consider call .fit() first.')

		probs = np.zeros((X.shape[0], self.class_count_.shape[0]))

		for i, x_k in enumerate(X):
			likelihood = self.A * np.exp((-1/2) * np.sum((x_k - self.theta_) * self.sigma_inv * (x_k - self.theta_), axis=1))

			posterior = likelihood * self.pw_
			sum_likelihood = np.sum(posterior)
			probs[i] = posterior/(sum_likelihood + 1e-5)

		return probs
