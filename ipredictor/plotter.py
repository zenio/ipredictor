#: -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
	"""
	Helper class designed for drawing prediction model results.
	Can draw interval-valued data as bars and scalar data as point graph.
	Just create an instance: plotter = Plotter()
	And add dataframes: plotter.add(dataframe)
	Show graphs: plotter.show()
	"""

	def __init__(self, title='', xlabel='', ylabel=''):
		"""
		:param title: common title of graph
		:param xlabel: x-axis label, printed below graph
		:param ylabel: y-axis label, printed at left of graph
		"""
		self.title = title
		self.xlabel = xlabel
		self.ylabel = ylabel

		self.fig, self.ax = plt.subplots()

		mpl.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif'})
		majorFmt = mpl.dates.DateFormatter('%Y-%m-%d')
		self.ax.xaxis.set_major_formatter(majorFmt)
		self.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=6))
		self.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
		self.ax.grid(alpha=0.4)
		self.ax.grid(which='minor', alpha=0.1)
		self.ax.grid(True)

	def add(self, data, **kwargs):
		"""
		Analyzes data and draws suitable graph
		:param data: interval-valued or scalar dataframe
		:param label: name of dataframe
		"""
		if isinstance(data['values'].iloc[0], np.ndarray):
			self._draw_intervals(data, **kwargs)
		else:
			self._draw_points(data, **kwargs)

	def _draw_points(self, data, **kwargs):
		"""
		Add scalar values graph to plot
		"""
		label = kwargs.get('label', '')
		color = kwargs.get('color', '')
		self.ax.plot(data, label=label, marker='o', markersize=3, alpha=0.6,
		                color=color)

	def _draw_intervals(self, data, **kwargs):
		"""Plots interval-valued data as bars
		:param data: dataframe containing interval-valued data
		:param label: name of interval-valued data
		:param color: matplotlib color identifier , default is autoselected
		"""
		label = kwargs.get('label', '')
		color = kwargs.get('color', '')
		heights = [x[1] for x in data['values']]
		bars = [x[0]-x[1] for x in data['values']]
		self.ax.bar(data.index, bars, 0.015, heights, alpha=0.2, label=label,
		            color=color, edgecolor=color)

	def show(self):
		"""
		Plots customized graph
		"""
		self.ax.legend(loc=3, prop={'size':10})
		self.fig.autofmt_xdate()
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		plt.title(self.title)
		plt.show()