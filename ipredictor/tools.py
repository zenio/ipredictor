#: -*- coding: utf-8 -*-
"""
	ipredictor.tools
	~~~~~~~~~~~~~~~~
	Helper and utility functions

	:license: see LICENSE for more details
"""
import pandas as pd
import numpy as np


def data_reader(filename, intervals=False, resample=False, sep=";"):
	"""Reads file specified by <filename> and returns data suitable for further
	analysis.
	Data should be interval-valued (2 columns with minimums and maximums) or
	presented as points (1 column). Firsth column should be either datestamp or
	index number.

	:param filename: data file path
	:param intervals: if true, then will read and return interval-valued data
	:param resample: if true, will resample missing rows in data
	:param sep: data file cells separator

	:return: formatted and resampled (if necessary) dataframe. If interval
			 valued data then 'minimums' & 'maximums' columns returned,
			 otherwise only 'values' column returnd for point data

	"""
	dtypes = {'values': np.float32, 'maxs': np.float32, 'mins': np.float32}
	date_parsers = ['datetime']
	headers = ['datetime']
	headers.extend(['mins', 'maxs'] if intervals else ['values'])

	data = pd.read_csv(filename, sep=sep, names=headers, dtype=dtypes,
	                   parse_dates=date_parsers)
	return data