#: -*- coding: utf-8 -*-
import logging

from ipredictor.models import HoltWintersI, ANNI, HybridI, LSTMI, HoltI, HoltWinters, ANN, HybridIPoints
from ipredictor import tools
from ipredictor.plotter import Plotter

import numpy as np

np.random.seed(7)


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)

	season_period = 24
	predict = season_period

	data = tools.data_reader('datasamples/25.csv', intervals=True, resample=True)
	train, test = data[:-predict], data[-predict:]

	model = HoltWintersI(train, season_period=season_period)
	prediction = model.predict(steps=predict)

	print "MSE: ", HoltWintersI.mse(test, prediction)
	print "MAPE: ", HoltWintersI.mape(test, prediction)
	print "ARV: ", HoltWintersI.arv(test, prediction, train)
	plotter = Plotter(rows=1, cols=1)

	plotter.add(train, color='b', label=u'Raw')
	plotter.add(test, color='g', label=u'Validation')
	plotter.add(prediction, color='r', label=u'Prediction')
	plotter.prepare(pos=0, title=u"", xlabel=u"Time", ylabel=u"Gas pressure")
	plotter.show()





