#: -*- coding: utf-8 -*-
from base import BasePredictModel, Prediction


class HoltWinters(BasePredictModel):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm.
	"""

	def __init__(self, data):
		"""
		:param data: initial training dataframe
		"""
		BasePredictModel.__init__(self)
		#: initial data
		self.X = data['values'].values

	def _predict(self):
		"""
		Creates prediction for specified steps ahead. Finally predict result
		will be compared with given sample data.
		:param sample: sample data used for comparison
		:param steps: predict steps ahead
		"""
		pass
