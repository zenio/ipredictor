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
			q: season period

		:param data: initial training dataframe
		:param season_period: data season period
		"""
		BasePredictModel.__init__(self, data)

		self.q = season_period
		#: 2 season data needed in order to prepare initial arrays
		if len(data) < 2 * self.q:
			raise ValueError

		self.L = []

	def _initialize_level_array(self):
		"""Fills initial values of level array"""
		self.L = [np.mean(self.X[:self.q], axis=0)]

	def _predict(self):
		"""Model prediction logic
		"""
		self._initialize_level_array()
