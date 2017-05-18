#: -*- coding: utf-8 -*-
"""
LSTM model implementation for interval-valued data
ToDo: add point data lstm model
"""
import numpy as np

from keras.layers import Dense, LSTM
from keras.models import Sequential

from .anni import ANNI


class LSTMI(ANNI):
	"""
	Same as ANN model. Except LSTM layer and neurons input data format
	"""

	def _build_model(self):
		"""
		ANN model is simple MLP with one hidden layer
		"""
		self.model = Sequential()
		self.model.add(LSTM(self.hidden_neurons*2,
		                    input_shape=(1, self.input_neurons)))
		self.model.add(Dense(self.output_neurons))
		self.model.compile(loss='mean_squared_error', optimizer='adam')

	def _generate_training_set(self):
		"""
		Should generate training sets for model fit procedure
		"""
		ANNI._generate_training_set(self)
		shape = self.trainingX.shape
		self.trainingX = np.reshape(self.trainingX, (shape[0], 1, shape[1]))

	def _reshape_sample(self, sample):
		"""
		Helper method that reshapes given data array into LSTM accetable form
		:param sample: data sample
		:return: reshaped array
		"""
		return np.reshape(sample, (1, 1, sample.shape[0]))