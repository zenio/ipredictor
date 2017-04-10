#: -*- coding: utf-8 -*-
"""
Time series prediction Artificial Neural Network model tests
"""
import unittest
import pandas as pd

from ipredictor.models import ANN


class ANNTestCase(unittest.TestCase):

	def setUp(self):
		self.season_period = 2
		self.values = range(1, self.season_period * 2+1)
		self.dataframe = pd.DataFrame.from_items([('values', self.values)])
		self.model = ANN(self.dataframe)

	def test_if_initial_user_data_is_scaled(self):
		self.assertTrue(0 <= self.model.X[-1] <= 1)

	def test_if_result_values_can_be_rescaled_back(self):
		self.model.Xf = self.model.X
		self.model._rescale_values()
		self.assertEqual(self.model.Xf[-1], self.values[-1])