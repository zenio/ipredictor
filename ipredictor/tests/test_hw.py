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
		self.values = range(1, self.season_period * 2+1)
		self.dataframe = DataFrame.from_items([('values', self.values)])
		self.model = HoltWinters(self.dataframe,
		                         season_period=self.season_period)
		self.model._init_starting_arrays()

	def test_if_data_length_compared_to_season_period(self):
		#: 2 seasons of data required for proper HW model initialisation
		self.assertRaises(ValueError, HoltWinters, self.dataframe,
		                  season_period=len(self.dataframe))

	def test_if_initial_level_array_properly_generated_before_predict(self):
		#: initial level is mean for 1 season
		self.assertEqual(self.model.L[0], 1.5)

	def test_if_trend_array_properly_generated(self):
		#: initial trend is pairwise  average of trend averages for two seasons
		self.assertEqual(self.model.T[0], 1)

	def test_if_initial_seasons_values_array_properly_generated(self):
		seasons = [-0.5, 0.5]
		self.assertSequenceEqual(self.model.S, seasons)

	def test_if_new_level_value_can_be_calculated(self):
		alpha = 0.5
		expected = 2
		#: new level formula: alpha * (current_value - last_season)
		#                       + alpha (last_trend + last_level)
		calculated_level = self.model._predict_level(0, alpha)
		self.assertTrue(np.isclose(expected, calculated_level, rtol=1e-03))

	def test_if_new_trend_level_value_properly_calculated(self):
		beta = 0.5
		expected = 1
		#: new trend formula: beta * (level_change) + (1 - beta) * prev_trend
		self.model.L.append(2.5)
		calculated_trend = self.model._predict_trend(beta)
		self.assertTrue(np.isclose(expected, calculated_trend, rtol=1e-03))

	def test_if_new_seasonal_coef_value_properly_calculated(self):
		gamma = 0.5
		expected = -1
		#: formula: gamma * (current_value - prev_trend - prev_level) +
		#:              (1 - gamma) * prev_season
		#: fake forecasts
		self.model.L.append(0)
		self.model.T.append(0)
		calculated_season = self.model._predict_seasonal(0, gamma)
		self.assertTrue(np.isclose(expected, calculated_season, rtol=1e-03))


