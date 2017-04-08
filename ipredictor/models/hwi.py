#: -*- coding: utf-8 -*-
import numpy as np

from .hw import HoltWinters
from .base import IntervalDataMixin

from ipredictor.defaults import INITIAL_COEF
from ipredictor.tools import flats_to_matrix


class HoltWintersI(IntervalDataMixin, HoltWinters):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm
	adapted for interval-valued data
	"""

	E = np.identity(2)

	def _check_initial_coefs(self, coefs):
		"""Set up initial coefficients
		:param coefs: list of coefs matrix [alpha, beta, gamma]
		:raises ValueError: if given coefs negative or greater that 1
		"""
		alpha, beta, gamma = coefs

		if any([alpha is not None and not isinstance(alpha, np.matrix),
		        beta is not None and not isinstance(beta, np.matrix),
		        gamma is not None and not isinstance(gamma, np.matrix)]):
			raise ValueError(u"Given coef matrix should be instance of "
			                 u"np.matrix")

		if any([not self.__is_correct_coefs_matrix(alpha),
		        not self.__is_correct_coefs_matrix(alpha),
		        not self.__is_correct_coefs_matrix(alpha)]):
			raise ValueError(u"All given matrix coefs values should be in "
			                 u"range [0;1]")


	def __is_correct_coefs_matrix(self, matrix):
		"""Checks if given matrix is filled with correct values in range [0;1]
		if specified.
		"""
		return matrix is None or (matrix >= 0).all() and (matrix <= 1).all()

	def _optimization_start_conditions(self):
		"""Override parents method that generates start conditions for
		optimization algorithm
		"""
		initial_values = np.array([INITIAL_COEF] * 12)
		boundaries = [(0, 1)] * 12
		return initial_values, boundaries

	def _retreive_coefs(self, coefs):
		"""Overriden coefs retreive function found by optimization algorithm"""
		self.alpha, self.beta, self.gamma = self._coefs =flats_to_matrix(coefs)

