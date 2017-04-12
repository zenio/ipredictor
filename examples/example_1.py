#: -*- coding: utf-8 -*-
import logging

from ipredictor.models import HoltWinters
from ipredictor.tools import data_reader


logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
	data = data_reader('../ipredictor/tests/assets/points.csv')
	model = HoltWinters(data, season_period=2)
	prediction = model.predict()