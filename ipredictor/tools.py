#: -*- coding: utf-8 -*-
"""
	ipredictor.tools
	~~~~~~~~~~~~~~~~
	Helper and utility functions

	:license: see LICENSE for more details
"""

def data_reader(filename, resample=True):
	"""Reads file specified by <filename> and returns data suitable for further
	analysis.
	:param filename: data file path
	:param resample: if true, will resample missing rows in data
	"""
	return True