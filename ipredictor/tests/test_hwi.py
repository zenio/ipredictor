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

	def test_if_new_trend_value_can_be_calculated(self):
		#: B is beta 2x2 matrix
		#: B * (level_change) + (1 - B) * prev_trend
		self.hwi.L.append(self.hwi._predict_level(0, self.A))
		expected = np.array([[0], [0]])
		calculated_trend = self.hwi._predict_trend(self.B)
		self.assertTrue(np.array_equal(expected, calculated_trend))

	def test_if_new_seasonal_factor_is_properly_calculated(self):
		#: G is gamma 2x2 matrix
		#: formula: G * (current_value - prev_trend - prev_level) +
		#:              (1 - G) * prev_season
		self.hwi.L.append(self.hwi._predict_level(0, self.A))
		self.hwi.T.append(self.hwi._predict_trend(self.B))
		expected = np.array([[-1.5], [-1.5]])
		calculated_season = self.hwi._predict_seasonal(0, self.G)
		self.assertTrue(np.array_equal(expected, calculated_season))

	def test_if_predefined_coefs_are_validated(self):
		bad = [1,2,3]
		self.assertRaises(ValueError, self.hwi._check_initial_coefs, bad)
		good = self.A, self.B, self.G
		try:
			self.hwi._check_initial_coefs(good)
		except ValueError:
			self.fail("Unexpected error raised")

	def test_if_auto_calculated_coefs_are_matrix(self):
		self.hwi.predict()
		self.assertIsInstance(self.hwi.alpha, np.matrix)
		self.assertIsInstance(self.hwi.beta, np.matrix)
		self.assertIsInstance(self.hwi.gamma, np.matrix)

	def test_if_one_step_prediction_calculated_as_it_should(self):
		expected = np.array([[3], [2]])
		self.hwi.coefs = self.A, self.B, self.G
		self.hwi.predict()
		self.assertTrue(np.array_equal(expected, self.hwi.Xf[0]))

	def test_if_rmse_can_be_calculated(self):
		result = self.hwi.predict(steps=4)
		try:
			HoltWintersI.rmse(self.dataframe, result)
		except ValueError:
			self.fail("Unexpected error raised")


