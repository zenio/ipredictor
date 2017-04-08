#: -*- coding: utf-8 -*-
import numpy as np

from .hw import HoltWinters


class HoltWintersI(HoltWinters):
	"""
	Model implements Holt-Winters exponetial smoothing predict algorithm
	adapted for interval-valued data
	"""

	E = np.identity(2)
