#: -*- coding: utf-8 -*-
"""
Prediction model based on Artificial Neural Network (ANN)
Simple MLP with one hidden layer.
"""
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from .base import BasePredictModel


class ANN(BasePredictModel):

	def __init__(self, data, **kwargs):
		BasePredictModel.__init__(self, data, **kwargs)

		self.scaler = MinMaxScaler(feature_range=(0, 1))
		self._scale_values()

	def _scale_values(self):
		"""
		Initial data should be scaled to values between 0 and 1 in order
		to be handled by neural network
		"""
		temp = np.array(self.X).reshape((len(self.X), 1))
		self.X = self.scaler.fit_transform(temp)

	def _rescale_values(self):
		"""
		Result values rescale function
		"""
		self.Xf = self.scaler.inverse_transform(self.Xf)