#: -*- coding: utf-8 -*-
from __future__ import division

import logging
import numpy as np

from base import BasePredictModel

from ipredictor.defaults import SEASON_PERIOD, INITIAL_COEF
from scipy.optimize import fmin_l_bfgs_b


logger = logging.getLogger(__name__)


class HoltWinters(BasePredictModel):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm.
	"""
	#: identity value
	E = 1

	def __init__(self, data, **kwargs):
		"""
		Properties:
			L: calculated level values array
			T: calculated trend values array
			S: calculated season coefs array
			q: season period

		:param data: initial training dataframe
		:param season_period: seasonal periodicity of data
		"""
		BasePredictModel.__init__(self, data, **kwargs)

		self.q = kwargs.get('season_period', SEASON_PERIOD)
		self._check_initials()

		self.L = []
		self.T = []
		self.S = []
		self.alpha = self.beta = self.gamma = None

	def _check_initials(self):
		"""User provied data length check
		"""
		#: 2 season data needed in order to prepare initial arrays
		if len(self.data) < 2 * self.q:
			raise ValueError("At least 2 season of data should be provided")

	@BasePredictModel.coefs.setter
	def coefs(self, value):
		"""Sets weights of model. Descendant model should validate if
		coefs are properly set
		"""
		self._check_initial_coefs(value)
		self._extract_coefs(value)

	def _check_initial_coefs(self, coefs):
		"""Set up initial coefficients
		:param alpha: level smooting coefficient
		:param beta: trend smooting coefficient
		:param gamma: seasonal factor smooting coefficient
		:raises ValueError: if given coefs negative or greater that 1
		"""
		alpha, beta, gamma = coefs
		if any([alpha is not None and (alpha > 1 or alpha < 0),
		        beta is not None  and (beta > 1 or beta < 0),
		        gamma is not None and (gamma > 1 or gamma < 0)]):
			raise ValueError(u"Given coef values should be in range [0;1]")

	def _extract_coefs(self, coefs):
		"""Unpacks coefs array into separate coefs matrixes"""
		self.alpha, self.beta, self.gamma = self._coefs = coefs

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
		"""Seasons array is average of level changes for first two seasons"""
		x = [self.X[i] - self.L[0] for i in range(self.q)]
		second_season_level = np.mean(self.X[self.q:self.q*2], axis=0)
		y = [self.X[self.q + i] - second_season_level  for i in range(self.q)]

		self.S = []
		for i in range(len(x)):
			self.S.append((x[i] + y[i]) / 2)
		self.S = [self.X[i] - self.L[0] for i in range(self.q)]

	def _init_starting_arrays(self):
		"""Helper method for initial arrays initialization
		"""
		self.Xf = []
		self.X = self.data['values'].values.tolist()
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

	def _predict(self, ignore_future=False):
		"""Model prediction logic
		:param ignore_future: if set does not forecast future values
		"""
		self._init_starting_arrays()

		total_loops = data_length = len(self.X)
		#: if steps provided, then should generate future values
		if self.steps and not ignore_future:
			total_loops = total_loops + self.steps - 1

		for step in range(0, total_loops):
			if step >= data_length:
				#: synthetic previous value
				self.X.append(self.Xf[-1])
			self.L.append(self._predict_level(step, self.alpha))
			self.T.append(self._predict_trend(self.beta))
			self.S.append(self._predict_seasonal(step, self.gamma))
			#: using forecasted seasonal factor from previous period
			self.Xf.append(np.array(self.L[-1] + self.T[-1] + self.S[-self.q]))

	def _optimization_start_conditions(self):
		"""Start conditions for optimization algo: 3 coefs"""
		initial_coefs = np.array([INITIAL_COEF] * 3)
		boundaries = [(0, 1)] * 3
		return initial_coefs, boundaries

	def _optimization_forecast(self, params):
		"""Performs prediction and returns error"""
		self._extract_coefs(params)
		self._predict(ignore_future=True)
		return self.rmse(self.X[1:], self.Xf)

	def _find_coefs(self):
		"""Automatically finds optimal coefs for model.
		Finds optimal smooting coefs which return lowest rmse value.
		"""
		log_level = logging.getLogger().getEffectiveLevel()
		iprint = 1 if log_level == logging.DEBUG else -1

		initial_coefs, boundaries = self._optimization_start_conditions()
		result = fmin_l_bfgs_b(self._optimization_forecast,
		                       x0=initial_coefs, bounds=boundaries,
		                       approx_grad=True, iprint=iprint,
		                       epsilon=1e-2)
		self._extract_coefs(result[0])
		logger.debug("Optimal coefficients found: {}".format(self._coefs))

	def _post_process_prediction(self):
		"""Post process prediction values. Just retrun result"""
		try:
			if not self.Xf[0].shape[0] == 2:
				self.Xf = [x[0] for x in self.Xf]
		except IndexError:
			pass
		return BasePredictModel._post_process_prediction(self)


