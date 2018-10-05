import sys

import numpy as np
import pandas as pd

np.random.seed(0)

class KCM_F_GHClustering:
	def __init__(self, c):
		self.c_ = c
	
	def fit(self, X):

		# Ensure that there are more samples than clusters.
		assert(X.shape[0] >= self.c_)

		# Initialization
		converged = True

		# Initialize clusters.
		initial_centers_idx = np.random.choice(np.arange(0, X.shape[0]), self.c_, replace=False)
		self.cluster_centers_ = X[initial_centers_idx, :]
		
		# Initial labels
		self.labels_ = np.zeros((X.shape[0], ), dtype=np.uint)

		while True:

			for j, x_k in enumerate(X):
				min_c = None
				min_dist = np.inf

				for c in self.c_:
					points_cluster = X[self.labels_ == c_]
					dist = self.distance_to_cluster_(x_k, points_cluster, inv_s_2)

					if dist < min_dist:
						min_dist = dist
						min_c = c

				# Assing point to the closest cluster.
				if self.labels_[j] != min_c:
					self.labels_[j] = min_c
					converged = False

			if converged:
				break

		return self
	
	def distance_to_cluster_(self, x_l, points_cluster, inv_s_2):
		P_i = points_cluster.shape[0]
		two_by_two_distance = 0

		for i in range(P_i):
			for j in range(P_i):
				two_by_two_distance += self.K_s_(points_cluster[i], points_cluster[j])

		return 1 - 2 * np.sum([self.K_s_(x_k, x_l, inv_s_2) for x_l in points_cluster])/P_i + two_by_two_distance/np.square(P_i)

	def K_s_(self, x_l, x_k, inv_s_2):

		# Ensure that the vectors have the same length.
		assert(x_l.shape == x_k.shape == inv_s_2.shape)

		return np.exp((-1/2) * np.sum(np.square(x_l - x_k) * inv_s_2))

def main(argv):
	pass

if __name__ == "__main__":
    main(sys.argv)