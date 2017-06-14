#: -*- coding: utf-8 -*-
from __future__ import division

import logging
import pandas as pd
import numpy as np

from abc import ABCMeta
from datetime import datetime

from ipredictor.defaults import RESAMPLE_PERIOD
from ipredictor.tools import dataframe_values_extractor


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
		return self.Xf[-self.steps:]

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
	@dataframe_values_extractor
	def rmse(real, predicted):
		"""Calculate and return rmse for interval valued data
		:param real: interval valued array
		:param predicted: predicted interval valued array
		:return: rmse result
		"""
		error = 0
		for i in range(0, len(real)):
			#: difference between previous forecast value and observed value
			mean = real[i] - predicted[i]
			error += np.dot(mean.transpose(), mean)[0][0]
		return float(error)

	@staticmethod
	@dataframe_values_extractor
	def mse(real, predicted):
		"""Interval mean square error MSEI accuracy measure method
		for interval-valued data
		"""
		error = 0
		fitted_intervals = min(len(real), len(predicted))
		for i in range(fitted_intervals):
			max_diff = real[i][0] - predicted[i][0]
			min_diff = real[i][1] - predicted[i][1]
			error += (max_diff**2 + min_diff**2)
		return float(error / (2 * fitted_intervals))

	@staticmethod
	@dataframe_values_extractor
	def mad(real, predicted):
		"""Interval mean absolute error MADI accuracy measure method
		for interval-valued data
		"""
		error = 0
		fitted_intervals = min(len(real), len(predicted))
		for i in range(fitted_intervals):
			max_diff = abs(real[i][0] - predicted[i][0])
			min_diff = abs(real[i][1] - predicted[i][1])
			error += (max_diff + min_diff)
		return float(error / (2 * fitted_intervals))

	@staticmethod
	@dataframe_values_extractor
	def arv(real, predicted):
		"""Interval average relative variance (ARVI) accuracy measure method
		for interval-valued data"""
		mse_max = 0
		mse_min = 0
		mse_avg_max = 0
		mse_avg_min = 0
		sample_mean = np.mean(real, axis=0)

		for i in range(min(len(real), len(predicted))):
			mse_max += (real[i][0] - predicted[i][0]) ** 2
			mse_min += (real[i][1] - predicted[i][1]) ** 2
			mse_avg_max += (real[i][0] - sample_mean[0]) ** 2
			mse_avg_min += (real[i][1] - sample_mean[1]) ** 2
		return float((mse_max + mse_min) / (mse_avg_max + mse_avg_min))

	@staticmethod
	@dataframe_values_extractor
	def mape(real, predicted):
		"""Mean absolute percentage error (MAPE) accuracy measure method
		for interval-valued data"""
		mape_h = 0
		mape_l = 0
		fitted_intervals = min(len(real), len(predicted))

		for i in range(fitted_intervals):
			mape_h += abs((real[i][0] - predicted[i][0]) / real[i][0])
			mape_l += abs((real[i][1] - predicted[i][1]) / real[i][1])
		mape_h = mape_h * 100 / fitted_intervals
		mape_l = mape_l * 100 / fitted_intervals
		return np.average([mape_h, mape_l])

	@staticmethod
	@dataframe_values_extractor
	def utale(real, predicted):
		"""Interval U of Theil statistics (UI) accuracy measure method
		for interval-valued data"""
		highs_diff = 0
		lows_diff = 0
		step_highs = 0
		step_lows = 0

		if len(real) < 2:
			raise ValueError(u"At least two real value required")

		for i in range(1, min(len(real), len(predicted))):
			highs_diff += (real[i][0] - predicted[i][0]) ** 2
			lows_diff += (real[i][1] - predicted[i][1]) ** 2
			step_highs += (real[i][0] - real[i-1][0]) ** 2
			step_lows += (real[i][1] - real[i-1][1]) ** 2
		diff_sum = highs_diff + lows_diff
		steps_sum = step_highs + step_lows
		return float(np.sqrt(diff_sum / steps_sum))

class Prediction(object):
	"""
	Model for results representation. Contains all necessary information for
	further data analysis, such as: RMSE, speed and prediction result itself.
	"""

	def __init__(self, result, rmse, elapsed_time):
		self.rmse = rmse
		self.elapsed_time = elapsed_time
		self.result = result