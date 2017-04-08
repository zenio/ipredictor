#: -*- coding: utf-8 -*-
"""
Package tools and utils tests
"""
import unittest

import numpy as np
import pandas as pd

from ipredictor.tools import data_reader


POINTS_DATA_FILE = 'assets/points.csv'
INTERVALS_DATA_FILE = 'assets/intervals.csv'


class DataReaderTestCase(unittest.TestCase):
	"""Data read and resample function tests"""

	def test_if_data_reader_returns_dataframe(self):
		dataframe = data_reader(POINTS_DATA_FILE)
		self.assertIsInstance(dataframe, pd.DataFrame)

	def test_if_data_properly_formatted(self):
		dataframe = data_reader(POINTS_DATA_FILE)
		value = dataframe['values'][0]
		self.assertEqual(type(value), np.float32)

	def test_if_can_read_intervals(self):
		dataframe = data_reader(INTERVALS_DATA_FILE, intervals=True)
		self.assertIn('values', dataframe)
		self.assertEqual(dataframe['values'].values[0].shape[1], 1)

	def test_if_data_is_resampled_when_necessary(self):
		resampled = data_reader(POINTS_DATA_FILE, resample=True)
		delta =  resampled.index[1] - resampled.index[0]
		self.assertEqual(delta.components.hours, 1)

