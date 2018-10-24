import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

class DataLoader:

	def __init__(self):
		self.le_ = LabelEncoder()

	def load(self, train_dir, test_dir):
		self.image_segmentation_ = pd.read_csv(train_dir, delimiter=',')
		self.image_segmentation_test_ = pd.read_csv(test_dir, delimiter=',')
		self.image_segmentation_ = shuffle(self.image_segmentation_)

	def data(self):
		X_train = self.image_segmentation_.drop("CLASS", axis=1).values
		y_train = self.image_segmentation_["CLASS"].copy().values

		X_test = self.image_segmentation_test_.drop("CLASS", axis=1).values
		y_test = self.image_segmentation_test_["CLASS"].copy().values

		
		y_train = self.le_.fit_transform(y_train)
		y_test = self.le_.transform(y_test)

		return X_train, y_train, X_test, y_test
