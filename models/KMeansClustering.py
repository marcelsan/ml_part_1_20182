import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)

class KMeansClustering:
	def __init__(self, c, tol=0.001, max_iter=300):
		self.c_ = c
		self.tol_ = tol
		self.max_iter_ = max_iter 

	def fit(self, X):

		# Ensure that there are more samples than clusters.
		assert(X.shape[0] >= self.c_)

		# Initialize clusters.
		initial_centers_idx = np.random.choice(np.arange(0, X.shape[0]), self.c_, replace=False)
		self.cluster_centers_ = X[initial_centers_idx, :]
		
		# Initial labels
		self.labels_ = np.zeros((X.shape[0], ), dtype=np.uint)

		for i in range(self.max_iter_):

			# Assign instance to appropriate cluster.
			for j, x_k in enumerate(X):
				distances_x = np.sum(np.square(self.cluster_centers_ - x_k), axis=1)
				cluster_idx = np.argmin(distances_x)
				self.labels_[j] = cluster_idx

			# Update centroid location.
			for c_ in range(self.c_):
				points = X[self.labels_ == c_]
				self.cluster_centers_[c_, :] = np.mean(points, axis=0)

		return self

	def predict(self, X):

		labels = np.zeros((X.shape[0], ), dtype=np.uint)

		for j, x_k in enumerate(X):
			distances_x = np.sum(np.square(self.cluster_centers_ - x_k), axis=1)
			labels[j] = np.argmin(distances_x)

		return labels

def main(argv):

	colors = 10*["g","r","c","b","k"]

	X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

	kmeans = KMeansClustering(c=3).fit(X)	

	for centroid in kmeans.cluster_centers_:
		plt.scatter(centroid[0], centroid[1], marker="o", color="k", s=150, linewidths=5)


	for i, classification in enumerate(kmeans.labels_):
		color = colors[classification]
		plt.scatter(X[i][0], X[i][1], marker="x", color=color, s=150, linewidths=5)

	plt.show()

if __name__ == "__main__":
    main(sys.argv)