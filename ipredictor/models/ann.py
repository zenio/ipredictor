#: -*- coding: utf-8 -*-
"""
Prediction model based on Artificial Neural Network (ANN)
Simple MLP with one hidden layer.
"""
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from .base import BasePredictModel


class ANN(BasePredictModel):

	def __init__(self, data, lookback=1, **kwargs):
		BasePredictModel.__init__(self, data, **kwargs)

		self.lookback = lookback
		self.trainingX = self.trainingY = None

		self.scaler = MinMaxScaler(feature_range=(0, 1))
		self._scale_values()

		#: experimentally found that for time series 2 season lookback
		#: is optimal and 2x input for hidden layer
		self.hidden_neurons = self.lookback * 4
		self.output_neurons = 1
		self.input_neurons = self.lookback * 2

		self._generate_training_set()

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

	def _check_initial_coefs(self, coefs):
		"""
		Validate if proper weights file provided. Model weights are saved to
		HDF5 format.
		"""
		if not coefs.endswith('.h5'):
			raise ValueError("Model weights should be HDF5 format")

	def _build_model(self):
		"""
		ANN model is simple MLP with one hidden layer
		"""
		self.model = Sequential()
		self.model.add(Dense(self.hidden_neurons,
		                     input_dim=self.hidden_neurons))
		self.model.add(Dense(self.output_neurons))

	def _generate_training_set(self):
		"""
		Should generate training sets for model fit procedure
		"""
		dataX, dataY = [], []
		for i in range(len(self.X)-self.lookback):
			a = self.X[i:(i+self.lookback), 0]
			dataX.append(a)
			dataY.append(self.X[i + self.lookback, 0])
		self.trainingX, self.trainingY = np.array(dataX), np.array(dataY)

