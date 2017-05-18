#: -*- coding: utf-8 -*-
"""
Holt interval-valued modification model tests
"""
from __future__ import division

import unittest
import numpy as np

from pandas import DataFrame

from ipredictor.models import HoltI
from ipredictor.tools import flats_to_matrix


class HoltITestCase(unittest.TestCase):
	"""Holt for interval-valued data model tests"""

	def setUp(self):
		self.periods = 2
		self.values = [np.array([[i+1], [i]]) for i in
		               range(1, self.periods+1)]
		self.dataframe = DataFrame.from_items([('values', self.values)])
		self.model = HoltI(self.dataframe)
		self.coefs = [0.5] * 8
		self.A, self.B = flats_to_matrix(self.coefs)
		self.model._init_starting_arrays()

	def test_if_initial_level_is_eqaul_to_first_provided_value(self):
		self.model._init_level_array()
		self.assertTrue(np.array_equal(self.model.L[0], self.model.X[0]))

	def test_if_initial_trend_is_equal_to_diff_between_two_first_elemnts(self):
		self.model._init_trend_array()
		diff = self.model.X[1] - self.model.X[0]
		self.assertTrue(np.array_equal(self.model.T[0], diff))

	def test_if_new_level_value_can_be_calculated(self):
		#: A is alpha 2x2 matrix
		#: new level formula: A * (current_value) + A (last_trend + last_level)
		expected = np.array([[2], [1]])
		calculated_level = self.model._predict_level(0, self.A)
		self.assertTrue(np.array_equal(expected, calculated_level))

	def test_if_new_trend_value_can_be_calculated(self):
		#: B is beta 2x2 matrix
		#: B * (level_change) + (1 - B) * prev_trend
		self.model.L.append(self.model._predict_level(0, self.A))
		expected = np.array([[0], [0]])
		calculated_trend = self.model._predict_trend(self.B)
		self.assertTrue(np.array_equal(expected, calculated_trend))

	def test_if_one_step_prediction_calculated_as_it_should(self):
		expected = np.array([[2], [1]])
		self.model.coefs = self.coefs
		self.model.predict()
		self.assertTrue(np.array_equal(expected, self.model.Xf[0]))

	def test_if_optimization_start_conditions_properly_settedup(self):
		inits, bounds = self.model._optimization_start_conditions()
		self.assertEqual(len(inits), 8)
		self.assertEqual(len(bounds), 8)