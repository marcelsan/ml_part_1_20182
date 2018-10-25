import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

class DataLoader:
	""" DataLoader, in charge of reading the dataset and encode the labels to numerical. """

	def __init__(self):
		self.le_ = LabelEncoder()

	def load(self, train_dir, test_dir):
		"""
			Load the training and testing dataset.
			
			Parameters
			---------
			train_dir : String
						Training dataset .CSV location.

			test_dir :  String
						Testing dataset .CSV location.
			
			Returns
			---------
			self: Object
		"""

		self.image_segmentation_ = pd.read_csv(train_dir, delimiter=',')
		self.image_segmentation_test_ = pd.read_csv(test_dir, delimiter=',')
		self.image_segmentation_ = shuffle(self.image_segmentation_)

		return self

	def data(self):
		"""
			Return the loaded training and testing data.

			Returns
			---------
			X_train : np.array([x0, x1, ...])
					 Matrix of training samples.
			y_train : np.array([y0, y1, ...])
					 Array of integer training labels.
			X_test :  np.array([x0, x1, ...])
					 Matrix of testing samples.
			y_test :  np.array([y0, y1, ...])
					 Array of integer testing labels

		"""
		
		X_train = self.image_segmentation_.drop("CLASS", axis=1).values
		y_train = self.image_segmentation_["CLASS"].copy().values

		X_test = self.image_segmentation_test_.drop("CLASS", axis=1).values
		y_test = self.image_segmentation_test_["CLASS"].copy().values

		# Encode the labels to numerical.
		y_train = self.le_.fit_transform(y_train)
		y_test = self.le_.transform(y_test)

		return X_train, y_train, X_test, y_test
