#: -*- coding: utf-8 -*-
from .hw import HoltWinters


class HoltWintersI(HoltWinters):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm
	adapted for interval-valued data
	"""
