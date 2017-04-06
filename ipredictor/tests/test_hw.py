#: -*- coding: utf-8 -*-
"""
Holt-Winters model tests
"""
import unittest
import numpy as np

from ipredictor.models import HoltWinters, Prediction
from ipredictor.tools import data_reader
from pandas import DataFrame


POINTS_DATA_FILE = 'assets/points.csv'


class HWTestCase(unittest.TestCase):
	"""Holt winters model tests"""

	def setUp(self):
		self.season_period = 2
		self.dataframe = data_reader(POINTS_DATA_FILE)
		self.model = HoltWinters(self.dataframe,
		                         season_period=self.season_period)
		self.data_mean = np.mean(self.dataframe['values'][:self.season_period],
		                         axis=0)

	def test_if_data_length_compared_to_season_period(self):
		#: 2 seasons of data required for proper HW model initialisation
		self.assertRaises(ValueError, HoltWinters, self.dataframe,
		                  season_period=len(self.dataframe))

	def test_if_initial_level_array_properly_generated_before_predict(self):
		self.model._init_level_array()
		#: initial level is mean for 1 season
		self.assertEqual(self.model.L[0], self.data_mean)

	def test_if_trend_array_properly_generated(self):
		#: initial trend is pairwise  average of trend averages for two seasons
		self.model._init_trend_array()
		trend_value = 0.19075
		self.assertTrue(np.isclose(self.model.T[0], trend_value, rtol=1e-03))

	def test_if_initial_seasons_values_array_properly_generated(self):
		self.model._init_level_array()
		self.model._init_seasons_array()
		seasons = [251.699 - self.data_mean, 250.578 - self.data_mean]
		self.assertTrue(np.isclose(self.model.S[0], seasons[0], rtol=1e-03))
		self.assertTrue(np.isclose(self.model.S[-1], seasons[-1], rtol=1e-03))

	def test_if_new_level_value_can_be_calculated(self):
		self.model._init_level_array()



