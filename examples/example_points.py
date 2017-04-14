#: -*- coding: utf-8 -*-
import logging

from ipredictor.models import HoltWinters
from ipredictor.tools import data_reader
from ipredictor.plotter import Plotter


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)

	data = data_reader('../ipredictor/tests/assets/points.csv')
	train, test = data[:-5], data[-5:]
	model = HoltWinters(train, season_period=2)
	prediction = model.predict(steps=5)

	plotter = Plotter()
	plotter.add(train, color='b')
	plotter.add(test, color='g')
	plotter.add(prediction, color='r')
	plotter.show()