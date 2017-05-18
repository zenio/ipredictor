#: -*- coding: utf-8 -*-
"""
Holt exponential smoothing model for interval-valued data.
Based on HoltWintersI model.
"""
from __future__ import division

import numpy as np

from ipredictor.tools import flats_to_matrix
from .hwi import HoltWintersI


class HoltI(HoltWintersI):
	"""HoltI differs from HoltWintersI model by lack of seasonal component
	"""

	def _check_initials(self):
		"""No data length check used
		"""
		if len(self.data) < 2:
			raise ValueError("At least 2 data values should be provided")

	def _check_initial_coefs(self, coefs):
		"""Set up initial coefficients
		:param coefs: list of coefs matrix [alpha, beta]
		:raises ValueError: if given coefs negative or greater that 1
		"""
		alpha, beta = flats_to_matrix(coefs)

		if any([alpha is not None and not isinstance(alpha, np.matrix),
		        beta is not None and not isinstance(beta, np.matrix)]):
			raise ValueError(u"Given coef matrix should be instance of "
			                 u"np.matrix")

		if any([not self._is_correct_coefs_matrix(alpha),
		        not self._is_correct_coefs_matrix(alpha)]):
			raise ValueError(u"All given matrix coefs values should be in "
			                 u"range [0;1]")

	def _extract_coefs(self, coefs):
		"""Unpacks coefs array into separate coefs matrixes"""
		self.alpha, self.beta = self._coefs = flats_to_matrix(coefs)

	def _init_level_array(self):
		"""Fills initial values of level array"""
		self.L = [self.X[0]]

	def _init_trend_array(self):
		"""Initial trend array is just difference of first two provided elemnts
		"""
		self.T = [self.X[1] - self.X[0]]

	def _init_seasons_array(self):
		"""Seasonal factors not used"""
		pass

	def _predict_level(self, step, A):
		"""Calculate level predict for given step.
		:param step: step for which level is calculated
		:param A: alpha smooting coefs matrix or value
		:return: calculated level
		"""
		return A * (self.X[step]) + (self.E - A) * (self.L[-1] + self.T[-1])

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
			#: using forecasted seasonal factor from previous period
			self.Xf.append(np.array(self.L[-1] + self.T[-1]))

	def _optimization_start_conditions(self):
		"""Override parents method that generates start conditions for
		optimization algorithm
		"""
		initial_values = np.array([0, 1, 0, 1, 0, 1, 0, 1])
		boundaries = [(0, 1)] * 8
		return initial_values, boundaries

