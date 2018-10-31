import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

np.random.seed(42)

class KCM_F_GHClustering:
	def __init__(self, c):
		self.c_ = c  # number of clusters.

	def fit(self, X):

		# Init X
		self.X_ = X

		# Initialize the 1/s^2 variable.
		self.init_inv_s2_()

		# Initialize clusters.
		centers_idx = np.random.choice(np.arange(0, X.shape[0]), self.c_, replace=False)
		clusters = centers_idx.reshape((self.c_, 1)).tolist()

		# Initial labels
		distances = self.distance_to_clusters_(clusters)
		self.labels_ = np.argmin(distances, axis=1)

		while True:
			clusters = self.get_clusters_(self.labels_)

			self.update_inv_s2_(clusters)
			
			distances = self.distance_to_clusters_(clusters)
			labels = np.argmin(distances, axis=1)

			# Assing point to the closest cluster.
			if np.all(self.labels_ == labels):
				break
			else:
				self.labels_ = labels

		return self

	def get_clusters_(self, labels):
		return [ np.where(labels == i)[0].tolist() for i in range(self.c_)]

	def update_inv_s2_(self, clusters):

		p = self.X_.shape[1]
		pi = np.zeros(p)

		# Calculate pi
		for j in range(p):
			for cluster in clusters:
				pi[j] += 1/(len(cluster)) * np.sum([self.kernel(self.X_[r], self.X_[s])*(self.X_[r][j] - self.X_[s][j]) ** 2 for (r, s) in itertools.product(cluster, cluster)])

		inv_s2 = self.inv_sigma_squared * np.power(np.prod(pi), 1/p)/pi

		self.inv_s2_ = inv_s2

	def distance_to_clusters_(self, clusters):
		'''
			Equation 21
		'''
		distances = np.zeros((self.X_.shape[0], self.c_))

		for i, x_k in enumerate(self.X_):
			for j, cluster in enumerate(clusters):
				
				# Size of the j-th cluster.
				Pj = len(cluster)

				distances[i, j] = 1 - 2 * np.sum([self.kernel(self.X_[l], x_k) for l in cluster])/ Pj + \
									np.sum([self.kernel(self.X_[r], self.X_[s]) for (r, s) in itertools.product(cluster, cluster)])/(Pj ** 2)


		return distances

	def init_inv_s2_(self):
		p = self.X_.shape[1]

		idx = np.arange(self.X_.shape[0])

		# Calculate the distance between each two sample of X.
		distances = [np.linalg.norm(self.X_[i] - self.X_[j]) for (i, j) in itertools.product(idx, idx) if i < j]

		# Calculate sigma squared.
		p10, p90 = np.percentile(distances, [10, 90])
		inv_sigma_squared = (p10 + p90)/2.0

		# Initialize inv_sigma2_
		self.inv_s2_ = np.ones(self.X_.shape[1]) * 1/inv_sigma_squared

		# Initialize 1/sigma^2
		self.inv_sigma_squared = (1/inv_sigma_squared)
	

	def kernel(self, x_l, x_k):
		# Ensure that the vectors have the same length.
		assert(x_l.shape == x_k.shape)

		return np.exp((-1/2) * np.sum(np.square(x_l - x_k) * self.inv_s2_))

def main(argv):
	X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

	kcm = KCM_F_GHClustering(c=3).fit(X)	

	colors = 10*["g","r","c","b","k"]

	for i, classification in enumerate(kcm.labels_):
		color = colors[classification]
		plt.scatter(X[i][0], X[i][1], marker="x", color=color, s=150, linewidths=5)

	plt.show()

if __name__ == "__main__":
    main(sys.argv)
