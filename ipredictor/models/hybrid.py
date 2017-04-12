#: -*- coding: utf-8 -*-
"""
Hybrid model. Uses HoltWinters model for linear component prediction and
ANN model for nonlinear prediction of estimates.
1. Creates HW prediction
2. Finds difference between input and prediction
3. Creates ANN forecast for estimates
4. Returns sum of two prediction results
"""
import numpy as np

from ipredictor.models import HoltWinters, HoltWintersI, ANN, ANNI
from .base import BasePredictModel


class Hybrid(BasePredictModel):
	"""
	Just a wrapper above HW and ANN classes with same inteface.
	Different linear and non-linear models can be provided by setter.
	For example: model = Hybrid(...)
				 model.non_linear_model = LSTM etc
	Just two limitations: model should share same interface and be
	appropropriate for given data
	"""
	def __init__(self, data, season_period=1, **kwargs):
		BasePredictModel.__init__(self, data, **kwargs)

		self.season_period = season_period
		#: non linear model will lookup for last two seasons of data
		self.lookback = self.season_period * 2

		#: by default this models used, if u want to change use setter
		self.is_intervals = False
		self._linear_model = HoltWinters
		self._non_linear_model = ANN

		if isinstance(self.X[0], (np.ndarray, list)):
			self.is_intervals = True
			self._linear_model = HoltWintersI
			self._non_linear_model = ANNI

	@property
	def linear_model(self):
		"""Returns current linear model used by hybrid model
		"""
		return self._linear_model

	@linear_model.setter
	def linear_model(self, cls):
		"""Sets new linear model used by hybrid model
		"""
		if self.is_intervals != cls.is_intervals:
			raise ValueError("Bad model for given data type")
		self._linear_model = cls

	@property
	def non_linear_model(self):
		"""Returns current non linear model used by hybrid model
		"""
		return self._non_linear_model

	@non_linear_model.setter
	def non_linear_model(self, cls):
		"""Sets new non linear model used by hybrid model
		"""
		if self.is_intervals != cls.is_intervals:
			raise ValueError("Bad model for given data type")
		self._non_linear_model = cls
