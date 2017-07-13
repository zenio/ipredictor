#: -*- coding: utf-8 -*-
"""
	ipredictor.tools
	~~~~~~~~~~~~~~~~
	Helper and utility functions

	:license: see LICENSE for more details
"""
from __future__ import division

import functools
import pandas as pd
import numpy as np

from defaults import RESAMPLE_PERIOD


def data_reader(filename, intervals=False, resample=None, sep=";",
                resample_period=RESAMPLE_PERIOD, **kwargs):
	"""Reads file specified by <filename> and returns data suitable for further
	analysis.
	Data should be interval-valued (2 columns with minimums and maximums) or
	presented as points (1 column). Firsth column should be presented as
	string timestamp

	:param filename: data file path
	:param intervals: if true, then will read and return interval-valued data
	:param resample: if true, will resample missing rows in data
	:param resample_period: identifier passed to resample function. Ex: H, S...
	:param sep: data file cells separator
	:param column: get only mins or maxs column of intervals

	:return: formatted and resampled (if necessary) dataframe. If interval
			 valued data then 'minimums' & 'maximums' columns returned,
			 otherwise only 'values' column returnd for point data
	"""
	dtypes = {'values': np.float32, 'maxs': np.float32, 'mins': np.float32}
	date_parsers = ['datetime']
	headers = ['datetime']
	headers.extend(['mins', 'maxs'] if intervals else ['values'])

	data = pd.read_csv(filename, sep=sep, names=headers, dtype=dtypes,
	                   parse_dates=date_parsers, index_col='datetime')
	if resample:
		data = data.resample(resample_period).mean().interpolate(method='time')

	if intervals:
		column = kwargs.get('column', None)
		if column:
			data['values'] = data[column]
		else:
			x = [np.array([[x], [y]]) for x,y in zip(data['maxs'],
			                                         data['mins'])]
			data['values'] = pd.Series.from_array(x, index=data.index)
		#: remove helper columns
		data = data.drop('mins', 1).drop('maxs', 1)
	return data

def flats_to_matrix(flats):
	"""Converts flat coefs to smooting matrix
	:param flats: 8 or 12 coefs to be used
	:return: 2 or 3 matrix array [alpha, beta] or [alpha, beta, gamma]
	"""
	result = []

	alphas = flats[:4]
	beta = flats[4:8]

	result.append(np.matrix([[alphas[0], alphas[1]], [alphas[2], alphas[3]]]))
	result.append(np.matrix([[beta[0], beta[1]], [beta[2], beta[3]]]))

	if len(flats) == 12:
		gamma = flats[8:12]
		result.append(np.matrix([[gamma[0], gamma[1]], [gamma[2], gamma[3]]]))

	return result

def validate_hdf5(filename):
	""":raises ValueError: if given filename is not in HDF5 format"""
	if not filename.endswith('.h5'):
		raise ValueError("Model weights should be HDF5 format")

def dataframe_values_extractor(func):
	"""Decorator function extracts dataframe values if dataframe passed
	:param func: decorated function
	:return: arrays instead of dataframes
	"""
	@functools.wraps(func)
	def f(*args, **kwargs):
		replaced_args = []
		for arg in args:
			try:
				#: test if data is dataset and retreive values if true
				arg = arg['values'].values
			except:
				pass
			replaced_args.append(arg)
		return func(*replaced_args, **kwargs)
	return f

def separate_components(dataframe):
	"""
	Separates linear and nonlinear components of given interval-valued
	dataframe
	:param dataframe: interval-valued dataframe
	:return: <dataframe>, <dataframe> - linear and nonlinear components
	"""
	vals = dataframe.values
	centers = [(x[0][0] + x[0][1]) / 2  for x in vals]
	radius = [(x[0][0] - x[0][1]) / 2  for x in vals]
	centers_df = pd.DataFrame.from_items([('values', centers)])
	radius_df = pd.DataFrame.from_items([('values', radius)])
	centers_df = centers_df.set_index(dataframe.index)
	radius_df = radius_df.set_index(dataframe.index)
	return centers_df, radius_df

def combine_components(centers, radius):
	"""
	Combines two separate linear and non-linear components into one interval-
	valued dataframe
	:param centers: linear component
	:param radius: non-linear component
	:return: combined interval-valued dataframe
	"""
	highs, lows = [], []
	for c, r in zip(centers.values, radius.values):
		highs.append(c + r)
		lows.append(c - r)
	x = [np.array([[x], [y]]) for x, y in zip(highs, lows)]
	combination = pd.DataFrame.from_items([('values', x)])
	combination = combination.set_index(centers.index)
	return combination