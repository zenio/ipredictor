#: -*- coding: utf-8 -*-
"""
Package tools and utils tests
"""
import unittest

import numpy as np
import pandas as pd

from ipredictor.tools import (data_reader, validate_hdf5, separate_components,
                              combine_components)


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

	def test_if_validates_hdf5_format(self):
		bad = 'badfile.txt'
		self.assertRaises(ValueError, validate_hdf5, bad)

		try:
			validate_hdf5('something.h5')
		except ValueError:
			self.fail("Unexpected error raised")

	def test_if_can_separate_intervals_into_components(self):
		values = [np.array([[i+1], [i]]) for i in range(1, 11)]
		df = pd.DataFrame.from_items([('values', values)])
		centers, radius = separate_components(df)
		self.assertTrue(all([len(centers) == 10, len(radius) == 10]))
		self.assertEqual(centers.values[0], 1.5)
		self.assertEqual(radius.values[0], 0.5)

	def test_if_can_compine_separate_components(self):
		centers = [np.array(i) for i in range(2, 12)]
		radius = [np.array(i) for i in range(1, 11)]
		centers_df = pd.DataFrame.from_items([('values', centers)])
		radius_df = pd.DataFrame.from_items([('values', radius)])
		combined = combine_components(centers_df, radius_df)

