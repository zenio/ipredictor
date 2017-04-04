#: -*- coding: utf-8 -*-
"""
Package tools and utils tests
"""
import unittest

from ipredictor.tools import data_reader


DATA_FILE = 'assets/points.csv'


class DataReaderTestCase(unittest.TestCase):
	"""Data read and resample function tests"""

	def test_if_can_read_data_file(self):
		self.assertIsNotNone(data_reader(DATA_FILE))