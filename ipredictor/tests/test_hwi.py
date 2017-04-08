#: -*- coding: utf-8 -*-
"""
Holt-Winters interval-valued modification model tests
"""
import unittest
import numpy as np

from ipredictor.models import HoltWintersI
from ipredictor.tools import flats_to_matrix
from pandas import DataFrame


class HWITestCase(unittest.TestCase):
	"""Holt winters for interval-valued data model tests"""

	def setUp(self):
		self.season_period = 2
		self.values = [np.array([[i+1], [i]]) for i in
							range(1, self.season_period * 2+1)]
		self.dataframe = DataFrame.from_items([('values', self.values)])
		self.hwi = HoltWintersI(self.dataframe,
		                        season_period=self.season_period)
		self.hwi._init_starting_arrays()
		self.A, self.B, self.G = flats_to_matrix([0.5] * 12)

	def test_if_initial_arrays_properly_generated_before_predict(self):
		#: level
		self.assertTrue(np.array_equal(np.array([[2.5], [1.5]]), self.hwi.L[0]))
		#: trend
		self.assertTrue(np.array_equal(np.array([[1], [1]]), self.hwi.T[0]))
		#: seasonal
		expected = [np.array([[-0.5], [-0.5]]), np.array([[0.5],  [0.5]])]
		self.assertTrue(np.array_equal(expected, self.hwi.S))

	def test_if_new_level_value_can_be_calculated(self):
		#: A is alpha 2x2 matrix
		#: new level formula: A * (current_value - last_season)
		#                       + A (last_trend + last_level)
		expected = np.array([[2.5], [1.5]])
		calculated_level = self.hwi._predict_level(0, self.A)
		self.assertTrue(np.array_equal(expected, calculated_level))




