#: -*- coding: utf-8 -*-
"""
Package tools and utils tests
"""
import unittest

import numpy as np
import pandas as pd

from datetime import datetime

from ipredictor.tools import data_reader


POINTS_DATA_FILE = 'assets/points.csv'


class DataReaderTestCase(unittest.TestCase):
	"""Data read and resample function tests"""

	def test_if_data_reader_returns_dataframe(self):
		dataframe = data_reader(POINTS_DATA_FILE)
		self.assertIsInstance(dataframe, pd.DataFrame)

	def test_if_data_properly_formatted(self):
		dataframe = data_reader(POINTS_DATA_FILE)
		dt = dataframe['datetime'][0]
		value = dataframe['values'][0]
		self.assertIsInstance(dt, datetime)
		self.assertEqual(type(value), np.float32)