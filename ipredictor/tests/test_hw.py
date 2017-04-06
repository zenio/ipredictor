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
		self.season_period = 24
		self.dataframe = data_reader(POINTS_DATA_FILE)
		self.model = HoltWinters(self.dataframe,
		                         season_period=self.season_period)

	def test_if_data_length_compared_to_season_period(self):
		#: 2 seasons of data required for proper HW model initialisation
		self.assertRaises(ValueError, HoltWinters, self.dataframe,
		                  season_period=len(self.dataframe))

	def test_if_initial_arrays_properly_generated_before_predict(self):
		self.model.predict(steps=10)
		#: initial level is mean for 1 season
		mean = np.mean(self.model.X[:self.season_period], axis=0)
		self.assertEqual(self.model.L[0], mean)



