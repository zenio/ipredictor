#: -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif'})
majorFmt = mpl.dates.DateFormatter('%Y-%m-%d')


class Plotter:
	"""
	Helper class designed for drawing prediction model results.
	Can draw interval-valued data as bars and scalar data as point graph.
	Just create an instance: plotter = Plotter()
	And add dataframes: plotter.add(dataframe)
	Show graphs: plotter.show()
	"""

	def __init__(self, rows=1, cols=1, total=None):
		"""
		:param title: common title of graph
		:param xlabel: x-axis label, printed below graph
		:param ylabel: y-axis label, printed at left of graph
		"""
		self.axs = []
		self.fig = plt.figure(facecolor='white')

		width_ratios = [1 if x < 2 else 1 for x in range(cols)]
		gs = mpl.gridspec.GridSpec(rows, cols, width_ratios=width_ratios)

		for x in range(rows * cols):
			ax = plt.subplot(gs[x])
			ax.grid(alpha=0.4)
			ax.grid(which='minor', alpha=0.1)
			ax.grid(True)
			self.axs.append(ax)
		plt.tight_layout(w_pad=0.5, h_pad=3.0)

	def add(self, data, **kwargs):
		"""
		Analyzes data and draws suitable graph
		:param data: interval-valued or scalar dataframe
		:param label: name of dataframe
		"""
		pos = kwargs.get('pos', 0)
		self.axs[pos].xaxis.set_major_formatter(majorFmt)

		if isinstance(data['values'].iloc[0], (np.ndarray, list)):
			self._draw_intervals(data, **kwargs)
		else:
			self._draw_points(data, **kwargs)

	def _draw_points(self, data, **kwargs):
		"""
		Add scalar values graph to plot
		:param data: dataframe containing interval-valued data
		:param label: name of dataseries
		:param color: matplotlib color identifier , default is autoselected
		:param pos: position in graphs
		"""
		label = kwargs.get('label', '')
		color = kwargs.get('color', '')
		pos = kwargs.get('pos', 0)
		self.axs[pos].plot(data, label=label, marker='o', markersize=3,
		                   alpha=0.6, color=color)

	def _draw_intervals(self, data, **kwargs):
		"""Plots interval-valued data as bars
		:param data: dataframe containing interval-valued data
		:param label: name of interval-valued data
		:param color: matplotlib color identifier , default is autoselected
		:param pos: position in graphs
		"""
		label = kwargs.get('label', '')
		color = kwargs.get('color', '')
		pos = kwargs.get('pos', 0)
		heights = [x[1] for x in data['values']]
		bars = [x[0]-x[1] for x in data['values']]
		self.axs[pos].bar(data.index, bars, 0.015, heights, alpha=0.5,
		                  label=label, color=color, edgecolor=color)

	def add_bars(self, bars, labels, pos=0, **kwargs):
		"""
		Draws bar with given labels
		"""
		x = range(len(bars))
		color = kwargs.get('color', ['r', 'g', 'b', 'c'])

		#self.axs[pos].bar(x, bars, color=color, align='center')
		#self.axs[pos].set_xticks(x, minor=False)
		#self.axs[pos].set_xticklabels(labels)

		self.axs[pos].barh(x, bars, color=color, align='center')
		self.axs[pos].set_yticks(x, minor=False)
		self.axs[pos].set_yticklabels(labels)

	def prepare(self, pos=0, title='', xlabel='', ylabel=''):
		"""
		Prepares customized graph
		"""
		self.axs[pos].legend(loc=3, prop={'size':10})
		#self.fig.autofmt_xdate()
		self.axs[pos].set_xlabel(xlabel)
		self.axs[pos].set_ylabel(ylabel)
		self.axs[pos].set_title(title)

	def show(self):
		"""Shows prepared figure"""
		plt.show()

	def save(self, name):
		"""Saves prepared figure
		:param name: imange name and path where image is saved
		"""
		plt.ioff()
		self.fig.set_size_inches(20, 10)
		self.fig.savefig(name, dpi=200, bbox_inches='tight')

	def add_table(self, cols, labels, pos=0):
		"""Adds one row information table on top side of figure
		:param cols: table columns data
		:labels: table header labels
		"""
		#: no need in title when table used
		self.axs[pos].set_title('')
		self.axs[pos].table(cellText=[cols], colLabels=labels, loc='top')

	def __del__(self):
		"""Clean figure"""
		plt.close(self.fig)