#: -*- coding: utf-8 -*-
"""
Holt-Winters interval-valued modification model tests
"""
import unittest
import numpy as np

from ipredictor.models import HoltWintersI
from pandas import DataFrame


class HWITestCase(unittest.TestCase):
	"""Holt winters for interval-valued data model tests"""

	def setUp(self):
		self.season_period = 2
		self.values = [np.array([i, i+1]) for i in range(1, self.season_period * 2+1)]
		self.dataframe = DataFrame.from_items([('values', self.values)])
		self.hwi = HoltWintersI(self.dataframe,
		                        season_period=self.season_period)

	def test_if_initial_arrays_properly_generated_before_predict(self):
		self.hwi._init_starting_arrays()
		#: level
		self.assertTrue(np.array_equal(np.array([1.5, 2.5]), self.hwi.L[0]))
		#: trend
		self.assertTrue(np.array_equal(np.array([1, 1]), self.hwi.T[0]))
		#: seasonal
		expected = [np.array([-0.5, -0.5]), np.array([ 0.5,  0.5])]
		self.assertTrue(np.array_equal(expected, self.hwi.S))





