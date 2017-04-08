#: -*- coding: utf-8 -*-
import itertools
import numpy as np

from base import BasePredictModel

from ipredictor.defaults import SEASON_PERIOD, INITIAL_COEF
from scipy.optimize import fmin_l_bfgs_b


class HoltWinters(BasePredictModel):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm.
	"""
	#: identity value
	E = 1

	def __init__(self, data, season_period=SEASON_PERIOD, **kwargs):
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

		self.q = season_period
		#: 2 season data needed in order to prepare initial arrays
		if len(data) < 2 * self.q:
			raise ValueError

		self.L = []
		self.T = []
		self.S = []
		self.alpha = self.beta = self.gamma = None

	@BasePredictModel.coefs.setter
	def coefs(self, value):
		"""Sets weights of model. Descendant model should validate if
		coefs are properly set
		"""
		self._check_initial_coefs(value)
		self.alpha, self.beta, self.gamma = self._coefs = value

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

	def _predict(self):
		"""Model prediction logic
		"""
		self._init_starting_arrays()

		total_loops = data_length = len(self.X)
		#: if steps provided, then should generate future values
		if self.steps:
			total_loops = total_loops + self.steps - 1

		for step in range(0, total_loops):
			if step >= data_length:
				#: synthetic previous value
				self.X = np.append(self.X, self.Xf[-1])
			self.L.append(self._predict_level(step, self.alpha))
			self.T.append(self._predict_trend(self.beta))
			self.S.append(self._predict_seasonal(step, self.gamma))
			#: using forecasted seasonal factor from previous period
			self.Xf.append(self.L[-1] + self.T[-1] + self.S[-self.q])

	def _optimization_start_conditions(self):
		"""Start conditions for optimization algo: 3 coefs"""
		initial_coefs = np.array([INITIAL_COEF] * 3)
		boundaries = [(0, 1)] * 3
		return initial_coefs, boundaries

	def _optimization_forecast(self, params):
		"""Performs prediction and returns error"""
		self._retreive_coefs(params)
		self._predict()
		return self._calculate_rmse(self.X[1:], self.Xf)

	def _find_coefs(self):
		"""Automatically finds optimal coefs for model.
		Finds optimal smooting coefs which return lowest rmse value.
		"""
		initial_coefs, boundaries = self._optimization_start_conditions()
		result = fmin_l_bfgs_b(self._optimization_forecast, factr=10.0,
		                       x0=initial_coefs, bounds=boundaries,
		                       approx_grad=True)
		self._retreive_coefs(result[0])

	def _retreive_coefs(self, coefs):
		"""Retreives coefs found by optimization function"""
		self.alpha, self.beta, self.gamma = self._coefs = coefs

