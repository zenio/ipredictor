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
		if isinstance(data['values'].iloc[0], (np.ndarray, list)):
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

	def prepare(self):
		"""
		Prepares customized graph
		"""
		self.ax.legend(loc=3, prop={'size':10})
		self.fig.autofmt_xdate()
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.ax.set_title(self.title)

	def show(self):
		"""Shows prepared figure"""
		plt.show()

	def save(self, name):
		"""Saves prepared figure
		:param name: imange name and path where image is saved
		"""
		plt.ioff()
		self.fig.savefig(name, dpi=200, figsize=(20,10))

	def add_table(self, cols, labels):
		"""Adds one row information table on top side of figure
		:param cols: table columns data
		:labels: table header labels
		"""
		#: no need in title when table used
		self.ax.set_title(self.title)
		self.ax.table(cellText=[cols], colLabels=labels, loc='top')

	def __del__(self):
		"""Clean figure"""
		plt.close(self.fig)