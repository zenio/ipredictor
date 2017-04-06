#: -*- coding: utf-8 -*-
import numpy as np

from base import BasePredictModel, Prediction

from ipredictor.defaults import SEASON_PERIOD


class HoltWinters(BasePredictModel):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm.
	"""
	#: identity value
	E = 1

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
		self.S = []

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

	def _init_starting_arrays(self):
		"""Helper method for initial arrays initialization
		"""
		self._init_level_array()
		self._init_trend_array()
		self._init_seasons_array()

	def _predict_level(self, step, A):
		"""Calculate level predict for given step.
		Previous season factor used for next step level calculation.
		:param step: step for which level is calculated
		:param A: alpha smooting coefs matrix or value
		:return: calculated level
		"""
		return A * (self.X[step] - self.S[-self.q]) + \
		       (self.E - A) * (self.L[-1] + self.T[-1])

	def _predict_trend(self, B):
		"""Calculate trend forecast for given step.
		Last found thend value found with _predict_level value used with
		penultimate step value.
		:param B: beta smooting coefs matrix or value
		:return: calculated trend matrix [max, min] value for next step
		"""
		return B * (self.L[-1] - self.L[-2]) + (self.E - B) * self.T[-1]

	def _predict_seasonal(self, step, G):
		"""Calculate seasonal factor forecast for given step
		:param step: step for which seasonal factor is calculated
		:param G: gamma smooting coefs matrix
		:return: calculated seasonal matrix [max, min] value for step
		"""
		return G * (self.X[step] - self.L[-2] - self.T[-2]) +\
		       (self.E - G) * self.S[-self.q]

	def _predict(self):
		"""Model prediction logic
		"""
		self._init_starting_arrays()

