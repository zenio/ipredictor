#: -*- coding: utf-8 -*-
"""
Hybrid model tests
"""
import unittest
import pandas as pd
import numpy as np

from ipredictor.models import HoltWinters, HoltWintersI, ANN, ANNI, Hybrid


class HybridTestCase(unittest.TestCase):

	def setUp(self):
		self.season_period = 2
		self.data_length = self.season_period * 4
		self.points = range(1, self.data_length + 1)
		self.scalar_df = pd.DataFrame.from_items([('values', self.points)])

		self.intervals = [np.array([[i+1], [i]]) for i in
		               range(1, self.data_length + 1)]
		self.interval_df = pd.DataFrame.from_items([('values', self.intervals)])

	def test_if_right_models_been_used_for_linear_and_nonlinear_parts(self):
		self.model = Hybrid(self.scalar_df, season_period=self.season_period)
		self.assertIs(self.model.linear_model, HoltWinters)
		self.assertIs(self.model.non_linear_model, ANN)
		self.model = Hybrid(self.interval_df, season_period=self.season_period)
		self.assertIs(self.model.linear_model, HoltWintersI)
		self.assertIs(self.model.non_linear_model, ANNI)

	def test_if_can_set_only_appropriate_model(self):
		self.model = Hybrid(self.scalar_df, season_period=self.season_period)
		with self.assertRaises(ValueError):
			self.model.linear_model = HoltWintersI
		with self.assertRaises(ValueError):
			self.model.non_linear_model = ANNI

	def test_if_values_can_be_predicted(self):
		self.model = Hybrid(self.scalar_df, season_period=self.season_period)
		self.assertIsNone(self.model.estimates)
		STEPS = 10
		result = self.model.predict(steps=STEPS)
		self.assertEqual(len(result), STEPS)
		self.assertIsNotNone(self.model.estimates)
		self.assertEqual(len(self.model.estimates), len(self.scalar_df)-1)