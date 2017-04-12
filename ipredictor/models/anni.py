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