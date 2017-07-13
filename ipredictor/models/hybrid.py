#: -*- coding: utf-8 -*-
"""
Hybrid model. Uses HoltWinters model for linear component prediction and
ANN model for nonlinear prediction of estimates.
1. Creates HW prediction
2. Finds difference between input and prediction
3. Creates ANN forecast for estimates
4. Returns sum of two prediction results
"""
import pandas as pd
import numpy as np

from itertools import izip

from ipredictor.models import HoltWinters, HoltWintersI, ANN, ANNI
from ipredictor.defaults import TRAIN_EPOCHS
from ipredictor.tools import separate_components, combine_components
from .base import BasePredictModel, IntervalDataMixin


class Hybrid(BasePredictModel):
	"""
	Just a wrapper above HW and ANN classes with same inteface.
	Different linear and non-linear models can be provided by setter.
	For example: model = Hybrid(...)
				 model.non_linear_model = LSTM etc
	Just two limitations: model should share same interface and be
	appropropriate for given data
	"""
	def __init__(self, data, **kwargs):
		BasePredictModel.__init__(self, data, **kwargs)

		self.season_period = kwargs.get('season_period', 1)
		#: non linear model will lookup for last two seasons of data
		self.lookback = self.season_period
		#: train epochs for nonlinear model
		self.train_epochs = kwargs.get('train_epochs', TRAIN_EPOCHS)

		#: by default this models used, if u want to change use setter
		self.is_intervals = False
		self._linear_model = HoltWinters
		self._non_linear_model = ANN
		self.estimates = None
		self.linear_predict = None
		self.non_linear_predict = None

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


	def _find_coefs(self):
		"""Nothing to find"""
		pass

	def _predict_linear_component(self):
		"""Predicts linear component"""
		linear = self._linear_model(self.data,
		                            season_period=self.season_period)
		self.linear_predict = linear.predict(steps=self.steps)
		return linear

	def _predict_non_linear_component(self):
		"""Predicts non linear component"""
		non_linear = self._non_linear_model(self.estimates,
		                                    lookback=self.lookback,
		                                    train_epochs=self.train_epochs)
		self.non_linear_predict = non_linear.predict(steps=self.steps)
		return non_linear

	def _predict(self):
		"""
		Predicts linear component with linear model. Finds estimates(errors) of
		prediction and passes to non linear model. Result is summ of linear and
		non linear model predictions.
		"""
		linear = self._predict_linear_component()

		real_values = self.X[1:]
		predicted_values = linear.Xf[:len(real_values)]
		diff = [x-y for x, y in izip(real_values, predicted_values)]

		self.estimates = pd.DataFrame.from_items([('values', diff)])
		self.estimates = self.estimates.set_index(self.data.index[1:])

		self._predict_non_linear_component()

		prediction = self.linear_predict + self.non_linear_predict
		self.Xf = prediction['values']

	def _post_process_prediction(self):
		"""Post process prediction values. Just retrun result"""
		return self.Xf.values.tolist()

class HybridI(IntervalDataMixin, Hybrid):
	pass

class HybridIPoints(IntervalDataMixin, BasePredictModel):
	"""
	Hybrid model for interval-valued data that uses another approach: separate
	intervals into linear and nonliear components and predict via "point" model
	"""
	def __init__(self, data, **kwargs):
		BasePredictModel.__init__(self, data, **kwargs)

		self.season_period = kwargs.get('season_period', 1)
		#: non linear model will lookup for last two seasons of data
		self.lookback = self.season_period
		#: train epochs for nonlinear model
		self.train_epochs = kwargs.get('train_epochs', TRAIN_EPOCHS)

		#: by default this models used, if u want to change use setter
		self.is_intervals = False
		self._linear_model = HoltWinters
		self._non_linear_model = ANN
		self.estimates = None
		self.linear_predict = None
		self.non_linear_predict = None
		self.linear_data, self.non_linear_data = separate_components(data)

	@property
	def linear_model(self):
		"""Returns current linear model used by hybrid model
		"""
		return self._linear_model

	@linear_model.setter
	def linear_model(self, cls):
		"""Sets new linear model used by hybrid model
		"""
		if cls.is_intervals:
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
		if cls.is_intervals:
			raise ValueError("Bad model for given data type")
		self._non_linear_model = cls

	def _find_coefs(self):
		"""Nothing to find"""
		pass

	def _predict_linear_component(self):
		"""Predicts linear component"""
		linear = self._linear_model(self.linear_data,
		                            season_period=self.season_period)
		self.linear_predict = linear.predict(steps=self.steps)
		return linear

	def _predict_non_linear_component(self):
		"""Predicts non linear component"""
		non_linear = self._non_linear_model(self.non_linear_data,
		                                    lookback=self.lookback,
		                                    train_epochs=self.train_epochs)
		self.non_linear_predict = non_linear.predict(steps=self.steps)
		return non_linear

	def _predict(self):
		"""
		Predicts linear component with linear model. Finds estimates(errors) of
		prediction and passes to non linear model. Result is summ of linear and
		non linear model predictions.
		"""
		self._predict_linear_component()
		self._predict_non_linear_component()

		prediction = combine_components(self.linear_predict,
		                                self.non_linear_predict)
		self.Xf = prediction['values']