#: -*- coding: utf-8 -*-
"""
	ipredictor.tools
	~~~~~~~~~~~~~~~~
	Helper and utility functions

	:license: see LICENSE for more details
"""
import pandas as pd
import numpy as np

from defaults import RESAMPLE_PERIOD


def data_reader(filename, intervals=False, resample=None, sep=";",
                resample_period=RESAMPLE_PERIOD):
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
	return data

def flats_to_matrix(flats):
	"""Converts 12 flat coefs to 3 smooting matrix
	:param flats: 12 coefs to be used
	:return: 3 matrix array [alpha, beta, gamma]
	"""
	alphas = flats[:4]
	betas = flats[4:8]
	gammas = flats[8:12]

	alpha = np.matrix([[alphas[0], alphas[1]], [alphas[2], alphas[3]]])
	beta = np.matrix([[betas[0], betas[1]], [betas[2], betas[3]]])
	gamma = np.matrix([[gammas[0], gammas[1]], [gammas[2], gammas[3]]])

	return alpha, beta, gamma