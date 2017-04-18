#: -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np

from abc import ABCMeta
from datetime import datetime

from ipredictor.defaults import RESAMPLE_PERIOD


logger = logging.getLogger(__name__)


class BasePredictModel(object):
	"""
	Abstract base predict model. Inherit from it in order to create your own
	prediction model.

	Properties:
		data: initial full dataframe
		X: initial values array
		Xf: predicted values array
		steps: number of preditction steps
		index: array of timestamps or other type of indexes
		coefs: coefficients, weights matrix and etc used by model logic

	:param resample_period: provide if different resample period is used
	"""
	__metaclass__ = ABCMeta

	is_intervals = False

	def __init__(self, data, **kwargs):
		self.elapsed_time = 0
		self.steps = 0
		self.data = data
		self.X = data['values'].values.tolist()
		self.Xf = []
		self.index = data.index
		self._coefs = None
		self.resample_period = kwargs.get('resample_period', RESAMPLE_PERIOD)

	def __repr__(self):
		return self.__class__.__name__

	def predict(self, steps=0):
		start_time = datetime.now()
		logger.debug("Started prediction routine at {}...".format(start_time))

		self.steps = steps
		if not self._coefs:
			logger.debug("Predefined coefficients not provided. Starting "
			             "search procedure...")
			self._find_coefs()
		self._predict()

		end_time = datetime.now()
		logger.debug("Ended prediction routine at {}...".format(end_time))
		self.elapsed_time = end_time - start_time
		logger.debug("Elapsed time: {}".format(self.elapsed_time))
		return self._result()

	def _predict(self):
		"""
		Descendant models should realazie their prediction algorithms logic in
		this method.
		As result should provide values of: rmse, elapsed time and result
		"""
		raise NotImplementedError("Please implement this method")

	def _result(self):
		"""
		Model prediction result.
		If data index is timestamp then future dates will be generated
		:return: Prediction instance with result
		"""
		prediction = self._post_process_prediction()
		result = pd.DataFrame.from_items([('values', prediction)])
		#: calculate future dates if necessary
		last_index = self.index[-1]
		if isinstance(last_index, pd.Timestamp):
			periods = self.steps + 1 if self.steps else 2
			dates = pd.date_range(last_index, periods=periods,
			                      freq=self.resample_period)
			result = result.set_index(dates[1:])
		return result

	@staticmethod
	def rmse(real, predicted):
		"""Calculate and return rmse for data forecasted values
		:param real: test values array
		:param predicted: predicted values array

		:return: rmse value
		"""
		try:
			#: test if data is dataset and retreive values if true
			real, predicted = real['values'], predicted['values']
		except:
			pass
		return np.sqrt((sum([(m - n) ** 2 for m, n in
		                     zip(real, predicted)])).mean())

	def _post_process_prediction(self):
		"""Post process prediction values"""
		return self.Xf[len(self.data)-1:]

	@property
	def coefs(self):
		"""Returns current coefs used by model
		Weights can be array as for HW model, or filepath as for ANN model, or
		something else.
		"""
		return self._coefs

	@coefs.setter
	def coefs(self, value):
		"""Sets weights of model. Descendant model should validate if
		coefs are properly set
		"""
		raise NotImplementedError("Please implement this method")

	def _find_coefs(self):
		"""Automatically find optimal coefs for model"""
		raise NotImplementedError("Please implement this method")

	def _check_initial_coefs(self, coefs):
		"""
		Coeficients and weights validation logic should be implemented in
		child classes
		"""
		raise NotImplementedError("Please implement this method")


class IntervalDataMixin:
	"""
	This mixin used to overrides default "point-valued" functions with
	interval-valued version
	"""
	is_intervals = True

	@staticmethod
	def rmse(real, predicted):
		"""Calculate and return rmse for interval valued data
		:param real: interval valued array
		:param predicted: predicted interval valued array
		:return: rmse result
		"""
		try:
			#: test if data is dataset and retreive values if true
			real = real['values'].values
			#predicted = predicted['values']
			predicted = predicted['values'].values
		except:
			pass

		error = 0
		for i in range(0, len(real)):
			#: difference between previous forecast value and observed value
			mean = real[i] - predicted[i]
			error += np.dot(mean.transpose(), mean)
		return error[0][0]


class Prediction(object):
	"""
	Model for results representation. Contains all necessary information for
	further data analysis, such as: RMSE, speed and prediction result itself.
	"""

	def __init__(self, result, rmse, elapsed_time):
		self.rmse = rmse
		self.elapsed_time = elapsed_time
		self.result = result