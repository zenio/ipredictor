#: -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from abc import ABCMeta


class BasePredictModel(object):
	"""
	Abstarct base predict model used by other predict models

	Properties:
		data: initial full dataframe
		X: initial values array
		Xf: predicted values array
		steps: number of preditction steps
		index: array of timestamps or other type of indexes
	"""
	__metaclass__ = ABCMeta

	def __init__(self, data):
		self.rmse = 0
		self.elapsed_time = 0
		self.steps = 1
		self.data = data
		self.X = data['values'].values
		self.index = data.index

	def predict(self, steps=1):
		self.steps = steps
		self._predict()
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
		Model prediction result return
		:return: Prediction instance with result
		"""
		return Prediction(self.data, self.rmse, self.elapsed_time)

	def _calculate_rmse(self, real, predicted):
		"""Calculate and return rmse for data forecasted values
		:param real: test values array
		:param predicted: predicted values array

		:return: rmse value
		"""
		return np.sqrt((sum([(m - n) ** 2 for m, n in
		                     zip(real, predicted)])).mean())


class Prediction(object):
	"""
	Model for results representation. Contains all necessary information for
	further data analysis, such as: RMSE, speed and prediction result itself.
	"""

	def __init__(self, result, rmse, elapsed_time):
		self.rmse = rmse
		self.elapsed_time = elapsed_time
		self.result = result