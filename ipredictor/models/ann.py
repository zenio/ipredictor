#: -*- coding: utf-8 -*-
"""
Prediction model based on Artificial Neural Network (ANN)
Simple MLP with one hidden layer.
"""
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from ipredictor.tools import validate_hdf5
from .base import BasePredictModel


class ANN(BasePredictModel):
	"""
	Prediction model based on Artificial Neural Network (ANN)
	Simple MLP with one hidden layer.
	Predefined weights can be provided by weights file location path.
	If no predefined coefs provided, then model automatically will find them
	and will set flag if weights been found.
	Calculated coefs can be saved by calling models: <save_coefs> method.
	"""

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
		self.input_neurons = self.lookback

		self._generate_training_set()
		self._build_model()

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
		validate_hdf5(coefs)

	def _build_model(self):
		"""
		ANN model is simple MLP with one hidden layer
		"""
		self.model = Sequential()
		self.model.add(Dense(self.hidden_neurons,
		                     input_dim=self.input_neurons))
		self.model.add(Dense(self.output_neurons))
		self.model.compile(loss='mean_squared_error', optimizer='adam')

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

	def _find_coefs(self):
		"""
		Starts training procedure and finds optimal weights for model
		"""
		self._coefs = None
		self.model.fit(self.trainingX, self.trainingY, epochs=1,
		               batch_size=200, verbose=1)
		#: set flag that coefs can be saved
		self._coefs = True

	def _predict(self):
		"""
		Prediction procedure. If predefined coefs not provided, then training
		dataset is used.
		"""
		self.Xf = self.model.predict(self.trainingX)
		self.Xf = np.reshape(self.Xf, (len(self.Xf), ))

		#: get last lookback data and predict one step ahead step by step
		sample = self.X[-self.lookback:, 0]
		for i in range(self.steps):
			reshaped_sample = np.reshape(sample, (1, sample.shape[0]))
			predicted_value = self.model.predict(reshaped_sample)
			self.Xf = np.append(self.Xf, predicted_value[0])
			#: use last prediction as input data
			sample = np.delete(sample, 0, axis=0)
			sample = np.append(sample, predicted_value)

	def _post_process_prediction(self):
		"""Overrides parent method.
		Rescales result value to real value
		"""
		self._rescale_values()
		return self.Xf[-self.steps:]

	def save_coefs(self, file):
		"""
		Saves calculated model weights to provided <file> in HDF5 format
		"""
		validate_hdf5(file)
		return self.model.save_weights(file)

	@BasePredictModel.coefs.setter
	def coefs(self, value):
		"""Sets weights of model. Weights is filename in HDF5 format used by
		model
		:param value: HDF5 file path
		"""
		validate_hdf5(value)
		self._coefs = value
		#self.model.load_weights(self._coefs)