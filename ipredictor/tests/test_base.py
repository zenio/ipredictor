#: -*- coding: utf-8 -*-
"""
Base model features tests
"""
from __future__ import division

import unittest
import numpy as np
import pandas as pd

from ipredictor.models import HoltWinters, Prediction
from ipredictor.models.base import IntervalDataMixin
from ipredictor.tools import data_reader


POINTS_DATA_FILE = 'assets/points.csv'


class BaseModelTestCase(unittest.TestCase):
	"""Base model tests"""

	def setUp(self):
		self.dataframe = data_reader(POINTS_DATA_FILE, resample=True)
		data_size = len(self.dataframe)
		train_size = int(data_size * 0.8)
		self.test_size = data_size - train_size
		self.train = self.dataframe [0:train_size:]
		self.test = self.dataframe[train_size:data_size:]
		self.model = HoltWinters(self.dataframe, season_period=2)

	def test_if_data_handled_properly(self):
		self.assertEqual(self.model.X[0], self.dataframe['values'][0])
		self.assertEqual(len(self.model.data), len(self.dataframe))
		self.assertEqual(self.model.index[0], self.dataframe.index[0])

	def test_if_rmse_calculated(self):
		result = HoltWinters.rmse(np.array([0, 1]), np.array([1, 2]))
		self.assertEqual(result, np.sqrt(2))
		result = HoltWinters.rmse(np.array([0, 0]), np.array([3, 3]))
		self.assertEqual(result, np.sqrt(18))

	def test_if_can_provide_own_predefined_weights(self):
		alpha = beta = gamma = 1
		coefs = [alpha, beta, gamma]
		self.model.coefs = coefs
		self.assertSequenceEqual(self.model.coefs, coefs)

	def test_if_prediction_dataframe_index_are_extrapolated(self):
		result = self.model.predict(steps=5)
		diff = self.dataframe.index[1] - self.dataframe.index[0]
		self.assertIsInstance(result.index[0], pd.Timestamp)
		self.assertEqual(result.index[0], self.dataframe.index[-1] + diff)

	def test_if_mse_error_for_its_properly_calculated(self):
		model = IntervalDataMixin()
		x = [np.array([[i+1], [i]]) for i in range(1, 11)]
		y = [np.array([[i+1], [i]]) for i in range(3, 13)]
		self.assertEqual(model.mse(x, y), 4)

	def test_if_mad_error_properly_calculated_for_its(self):
		model = IntervalDataMixin()
		x = [np.array([[i+1], [i]]) for i in range(1, 11)]
		y = [np.array([[i+1], [i]]) for i in range(3, 13)]
		self.assertEqual(model.mad(x, y), 2)

	def test_if_both_array_and_dataframe_can_be_used_for_error_calc(self):
		model = IntervalDataMixin()
		x = [np.array([[i+1], [i]]) for i in range(1, 11)]
		y = [np.array([[i+1], [i]]) for i in range(3, 13)]
		xdf = pd.DataFrame.from_items([('values', x)])
		ydf = pd.DataFrame.from_items([('values', y)])

		try:
			model.mse(x,y)
			model.mse(xdf,ydf)
		except ValueError:
			self.fail("Unexpected error raised")

	def test_if_arv_error_value_properly_calculated(self):
		model = IntervalDataMixin()
		x = [np.array([[i+3], [i+2]]) for i in range(5)]
		y = [np.array([[i+1], [i+1]]) for i in range(5)]
		self.assertEqual(model.arv(x, y), 1.25)