#: -*- coding: utf-8 -*-
"""
Interval-valued data prediction model based on Artificial Neural Network (ANN)
"""
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from ann import ANN
from .base import IntervalDataMixin


class ANNI(IntervalDataMixin, ANN):
	"""
	Interval-valued prediction model based on Artificial Neural Network (ANN)
	Simple MLP with one hidden and one output layers.
	Output layer contains two neurons for both interval ends [max;min]
	"""

	def _configure_neurons(self):
		"""Overriden ANN method. Configures MLP neurons amount
		Input neurons amount increased because inteval-valued data is two times
		bigger.
		Output neurons amount is 2: for miniumum and maximum value of interal
		"""
		self.input_neurons = self.lookback * 2
		self.hidden_neurons = self.input_neurons * 4
		self.output_neurons = 2

	def _preprocess_values(self):
		"""
		Interval-valued data should be flattened because MLP doesn't support
		complex inputs
		[[2,1],[3,2]] -> [2,1,3,2]
		"""
		mixed = []
		for i in range(len(self.X)):
			mixed.append(self.X[i][0])
			mixed.append(self.X[i][1])
		self.X = np.array(mixed)