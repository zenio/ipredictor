#: -*- coding: utf-8 -*-
"""
Interval-valued data prediction Artificial Neural Network model tests
"""
import unittest
import pandas as pd

from ipredictor.models import ANNI


class ANNITestCase(unittest.TestCase):

	def setUp(self):
		self.lookback = 2
		self.values = range(1, self.lookback * 4 + 1)
		self.dataframe = pd.DataFrame.from_items([('values', self.values)])
		self.model = ANNI(self.dataframe, lookback=self.lookback)

	def test_if_neurons_amount_properly_configured(self):
		self.assertEqual(self.model.input_neurons, self.lookback * 2)
		self.assertEqual(self.model.hidden_neurons, self.lookback * 8)
		self.assertEqual(self.model.output_neurons, 2)