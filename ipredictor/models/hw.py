#: -*- coding: utf-8 -*-
import numpy as np

from base import BasePredictModel, Prediction

from ipredictor.defaults import SEASON_PERIOD


class HoltWinters(BasePredictModel):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm.
	"""

	def __init__(self, data, season_period=SEASON_PERIOD):
		"""
		Properties:
			L: calculated level values array
			T: calculated trend values array
			S: calculated season coefs array
			q: season period

		:param data: initial training dataframe
		:param season_period: seasonal periodicity of data
		"""
		BasePredictModel.__init__(self, data)

		self.q = season_period
		#: 2 season data needed in order to prepare initial arrays
		if len(data) < 2 * self.q:
			raise ValueError

		self.L = []
		self.T = []

	def _init_level_array(self):
		"""Fills initial values of level array"""
		self.L = [np.mean(self.X[:self.q], axis=0)]

	def _init_trend_array(self):
		"""
		Trend array initial is pairwise  average of trend averages across
		two seasons divided by square season period
		"""
		self.T = [sum([self.X[i + self.q] - self.X[i]
		               for i in range(self.q)]) / (self.q ** 2)]

	def _init_seasons_array(self):
		"""Seasons array is calculated from first season of provided data and
		previously calculated level value"""
		self.S = [self.X[i] - self.L[0] for i in range(self.q)]

	def _predict(self):
		"""Model prediction logic
		"""
		self._init_level_array()
		self._init_trend_array()
		self._init_seasons_array()
