#: -*- coding: utf-8 -*-
"""
Base model features tests
"""
import unittest
import numpy as np

from ipredictor.models import HoltWinters, Prediction
from ipredictor.tools import data_reader
from pandas import DataFrame


POINTS_DATA_FILE = 'assets/points.csv'


class BaseModelTestCase(unittest.TestCase):
	"""Base model tests"""

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
		self.assertEqual(len(self.model.data), len(self.dataframe))
		self.assertEqual(self.model.index[0], self.dataframe.index[0])

	def test_if_rmse_calculated(self):
		result = self.model._calculate_rmse(np.array([0, 1]), np.array([1, 2]))
		self.assertEqual(result, np.sqrt(2))
		result = self.model._calculate_rmse(np.array([0, 0]), np.array([3, 3]))
		self.assertEqual(result, np.sqrt(18))

	def test_if_can_provide_own_predefined_weights(self):
		alpha = beta = gamma = 1
		coefs = [alpha, beta, gamma]
		self.model.coefs = coefs
		self.assertSequenceEqual(self.model.coefs, coefs)

	def test_if_coefs_automatically_been_calculated_if_not_provided(self):
		self.assertIsNone(self.model.coefs)
		self.model.predict()
		self.assertIsNotNone(self.model.coefs)