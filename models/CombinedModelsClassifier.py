import math
import sys

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from models.BayesianClassifier import BayesianClassifier
from models.KNeighborsClassifier import KNNClassifier

class CombinedModelsClassifier(BaseEstimator, ClassifierMixin):

	"""
		This classifier combines the KNNClassifier and the BayesianClassifier. It also
		uses two views (two sets of columns) of the dataset plus the entire dataset
		(that is, all columns).
		To combine the classifiers and obtain the resulting classification it uses the following
		affectation equation:

		argmax [(1-L) P(w) + L max(Pgauss_view1(w|x), Pgauss_view2(w|x), Pgauss_allcolumns(w|x),
		                          Pknn_view1(w|x), Pknn_view2(w|x), Pknn_allcolumns(w|x))]

		where L is the numer of views, in this case L = 3.

		Parameters
		---------
		n_neighbors : Int
					  The number of neighbors for the KNN Classifier.

		view1_columns : List of Int 
						The columns indexes for the first view of the dataset.
		
		view2_columns : List of Int
						The columns indexes for the second view of the dataset.
	"""

	def __init__(self, n_neighbors, view1_columns, view2_columns):
		self.name_ = "CombinedModelsClassifier"

		# Number of views
		self.L = 3

		# Initialize the models.
		self.gauss_view1_ = BayesianClassifier() # View 1.
		self.gauss_view2_ = BayesianClassifier() # View 2.
		self.gauss_view3_ = BayesianClassifier() # Complete dataset.

		self.knn_view1_ = KNNClassifier(n_neighbors=n_neighbors) # View 1.
		self.knn_view2_ = KNNClassifier(n_neighbors=n_neighbors) # View 2.
		self.knn_view3_ = KNNClassifier(n_neighbors=n_neighbors) # Complete dataset.

		# Store the columns index for each view.
		self.view1_columns_ = view1_columns
		self.view2_columns_ = view2_columns

	def fit(self, X, y, **kwargs):
		"""
		Fit the classifiers based on X and y.

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

		if kwargs:
			self.view1_columns_ = kwargs['view1_columns']
			self.view2_columns_ = kwargs['view2_columns']
			self.knn_view1_.n_neighbors = kwargs['n_neighbors']
			self.knn_view2_.n_neighbors = kwargs['n_neighbors']
			self.knn_view3_.n_neighbors = kwargs['n_neighbors']

		X_view1 = X[:, self.view1_columns_]
		X_view2 = X[:, self.view2_columns_]

		# Train the KNN classifiers.
		self.knn_view1_.fit(X_view1, y)
		self.knn_view2_.fit(X_view2, y)
		self.knn_view3_.fit(X, y)

		# Train the Bayesian classifiers.
		self.gauss_view1_.fit(X_view1, y)
		self.gauss_view2_.fit(X_view2, y)
		self.gauss_view3_.fit(X, y)

		# Calculate the probabilities.
		_, classes_count = np.unique(y, return_counts=True)
		self.P_w = classes_count/np.sum(classes_count)

		return self

	def predict(self, X):
		"""
		Perform classification on an array of test vectors X. It uses the affectation formula 
		below to obtain the final classification.

		argmax [(1-L) P(w) + L max(Pgauss_view1(w|x), Pgauss_view2(w|x), Pgauss_allcolumns(w|x),
		                          Pknn_view1(w|x), Pknn_view2(w|x), Pknn_allcolumns(w|x))]

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

		X_view1 = X[:, self.view1_columns_]
		X_view2 = X[:, self.view2_columns_]

		combined = (1-self.L) * self.P_w + self.L * np.max([self.gauss_view1_.predict_proba(X_view1),
															self.gauss_view2_.predict_proba(X_view2),
															self.gauss_view3_.predict_proba(X),
															self.knn_view1_.predict_proba(X_view1),
															self.knn_view2_.predict_proba(X_view2),
															self.knn_view3_.predict_proba(X)], axis=0)

		return np.argmax(combined, axis=1)

		