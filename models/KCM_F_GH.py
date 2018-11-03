import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

np.random.seed(42)

class KCM_F_GHClustering:
	"""
	Implements KCM-F-GH algorithm proposed in 'Gaussian kernel c-means hard 
	clustering algorithms with automated computation of the width 
	hyper-parameters'.

	Reference:
	---------
	DE CARVALHO, Francisco de AT et al. Gaussian kernel c-means hard clustering algorithms 
	with automated computation of the width hyper-parameters. 
	Pattern Recognition, v. 79, p. 370-386, 2018.
	
	Available at: https://www.sciencedirect.com/science/article/pii/S0031320318300712
	"""

	def __init__(self, c, verbose=True):
		"""
		Parameters
		---------
		c : Int
			Number of clusters.

		verbose: bool, default: False
			Option for producing detailed logging information.
			If True, it enables verbose output.
		"""

		self.c_ = c  # number of clusters.
		self.verbose_ = verbose 

	def fit(self, X, y=None):
		"""
		Implement the KCM-F-GH algorithm and store the obtained clusters.

		Parameters
		---------
		X : shape (n_samples, n_features)
			Training vectors, where n_samples is the number of training samples
			and n_features is the number of features.

		Returns
		---------
		self : object
		"""

		# Init the internal X variable.
		self.X_ = X

		# Initialize the 1/s^2 variable (the width parameter).
		self.init_inv_s2_()

		# Initialize clusters.
		centers_idx = np.random.choice(np.arange(0, X.shape[0]), self.c_, replace=False)
		clusters = centers_idx.reshape((self.c_, 1)).tolist()

		# Initial labels
		distances = self.distance_to_clusters_(self.X_, clusters)
		self.labels_ = np.argmin(distances, axis=1)

		iterations = 0

		if self.verbose_:
			print("Running iteration: ", end='', flush=True)
		
		while True:
			iterations += 1

			if self.verbose_:
				print("%d " %(iterations), end='', flush=True)

			# Obtain the clusters from the labels.
			self.clusters = self.build_clusters_(self.labels_)

			# Update the width parameters.
			self.update_inv_s2_(self.clusters)
			
			# Representation step. Compute the distances of each element for all clusters.
			distances = self.distance_to_clusters_(self.X_, self.clusters)
			
			# Allocation step. Assign each element for the closest cluster.
			labels = np.argmin(distances, axis=1)

			# Stop condition.
			if np.all(self.labels_ == labels):
				break
			
			self.labels_ = labels

		if self.verbose_:
			print()
			print("Total number of iterations until converge: %d" %(iterations))

		return self

	def predict(self, X):
		distances = self.distance_to_clusters_(X, self.clusters)
		
		self.cost_ = np.sum(distances)

		# Allocation step. Assign each element for the closest cluster.
		labels = np.argmin(distances, axis=1)

		return labels

	def build_clusters_(self, labels):
		'''
			Build clusters representation from labels.
		'''
		return [ np.where(labels == i)[0].tolist() for i in range(self.c_)]

	def update_inv_s2_(self, clusters):
		"""
		Computation of the width parameters for each dimension.

		Parameters
		---------
		clusters : list of list of Int.
				   Each i-element of list is a list of elements that belongs to the cluster
				   index by i.

				   Example:
				   [[1,2], [3], [4, 5]]

				   In this case, the elements 1 and 2 belongs to cluster 0, the element 3 belongs to 
				   cluster 1 and so on.

		"""

		# Number of features of the training data.
		p = self.X_.shape[1]

		# Initialize the pi term.
		pi = np.zeros(p)

		# Calculate the pi term
		for j in range(p):
			for cluster in clusters:
				# Size of the cluster.
				P = len(cluster)

				# Find all combinations of elements of the cluster.
				pairs = itertools.product(cluster, cluster)

				# Calculate the pi_j (for j = 1...p)
				pi[j] += 1/P * np.sum([self.kernel_(self.X_[r], self.X_[s])*(self.X_[r][j] - self.X_[s][j]) ** 2 for (r, s) in pairs if r < s])

		# Evaluate the equation 24.
		self.inv_s2_ = self.inv_sigma_squared * np.power(np.prod(pi), 1/p)/pi

	def distance_to_clusters_(self, X, clusters):
		"""
		Evaluate the distance of each element on training set to all clusters.

		Parameters
		---------
		clusters : list of list of Int.
				   Each i-element of list is a list of elements that belongs to the cluster
				   index by i.

				   Example:
				   [[1,2], [3], [4, 5]]

				   In this case, the elements 1 and 2 belongs to cluster 0, the element 3 belongs to 
				   cluster 1 and so on.
		"""
		
		distances = np.zeros((X.shape[0], self.c_))

		for j, cluster in enumerate(clusters):
			
			# Size of the j-th cluster.
			Pj = len(cluster)

			# Find all combinations of elements of the cluster.
			pairs = itertools.product(cluster, repeat=2)

			# Evaluate the second term of equation 21. Calculate the K(x_r, x_s) for all combination (x_r, x_s) 
			# of elements of the cluster.
			sum_kernel_xr_xs = np.sum([self.kernel_(self.X_[r], self.X_[s]) for (r, s) in pairs if r < s])

			for i, x_k in enumerate(X):
				# Evaluate the first term of equation 21. Calculate the K(x_k, x_l) for all elements x_l of the cluster.
				sum_kernel_xk_xl = np.sum([self.kernel_(self.X_[l], x_k) for l in cluster])

				# Evaluate the full equation (equation 21).
				distances[i, j] = 1 - 2 * sum_kernel_xk_xl/Pj + sum_kernel_xr_xs/(Pj ** 2)

		return distances

	def init_inv_s2_(self):
		p = self.X_.shape[1]

		idx = np.arange(self.X_.shape[0])

		# Calculate the distance between each two sample of X.
		distances = [np.linalg.norm(self.X_[i] - self.X_[j]) for (i, j) in itertools.product(idx, idx) if i < j]

		# Calculate the inverse sigma squared.
		p10, p90 = np.percentile(distances, [10, 90])
		sigma_squared = 0.5 * (p10 + p90)

		# Initialize 1/s^2
		self.inv_s2_ = np.ones(self.X_.shape[1]) * 1/sigma_squared

		# Initialize 1/sigma^2
		self.inv_sigma_squared = (1/sigma_squared)
	
	def kernel_(self, x_l, x_k):
		"""	
		Compute the Gaussian kernel. Where inv_s2_ is the width parameter.
	
		Parameters
		---------
		x_l, x_k : numpy.array of shape (n_features, ).

		"""

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
