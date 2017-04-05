#: -*- coding: utf-8 -*-
"""
Holt-Winters models tests
"""
import unittest
import numpy as np

from ipredictor.models import HoltWinters, Prediction
from ipredictor.tools import data_reader
from pandas import DataFrame


POINTS_DATA_FILE = 'assets/points.csv'


class HwModelTestCase(unittest.TestCase):
	"""Holt-Winters exponential smooting model tests"""

	def setUp(self):
		self.dataframe = data_reader(POINTS_DATA_FILE)
		data_size = len(self.dataframe)
		train_size = int(data_size * 0.8)
		self.test_size = data_size - train_size
		self.train = self.dataframe [0:train_size:]
		self.test = self.dataframe[train_size:data_size:]
		self.model = HoltWinters(self.dataframe)

	def test_if_data_handled_properly(self):
		self.assertEqual(self.model.X[0], self.dataframe['values'][0])

	def test_if_predict_returns_result(self):
		predict = self.model.predict(self.test, steps=self.test_size)
		self.assertIsInstance(predict, Prediction)
		self.assertIsNotNone(predict.rmse)
		self.assertIsNotNone(predict.elapsed_time)
		self.assertIsInstance(predict.result, DataFrame)

	def test_if_raises_error_on_when_predict_steps_greater_than_data_len(self):
		self.assertRaises(ValueError, self.model.predict, self.test,
		                  steps=len(self.test)+1)

	def test_if_rmse_calculated(self):
		result = self.model._calculate_rmse(np.array([0, 1]), np.array([1, 2]))
		self.assertEqual(result, np.sqrt(2))
		result = self.model._calculate_rmse(np.array([0, 0]), np.array([3, 3]))
		self.assertEqual(result, np.sqrt(18))