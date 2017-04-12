#: -*- coding: utf-8 -*-
"""
Interval-valued data prediction Artificial Neural Network model tests
"""
import unittest
import pandas as pd
import numpy as np

from ipredictor.models import ANNI


class ANNITestCase(unittest.TestCase):

	def setUp(self):
		self.lookback = 2
		self.data_length = self.lookback * 4
		self.values = [np.array([[i+1], [i]]) for i in
		               range(1, self.data_length + 1)]
		self.dataframe = pd.DataFrame.from_items([('values', self.values)])
		self.model = ANNI(self.dataframe, lookback=self.lookback)

	def test_if_neurons_amount_properly_configured(self):
		self.assertEqual(self.model.input_neurons, self.lookback * 2)
		self.assertEqual(self.model.hidden_neurons, self.lookback * 8)
		self.assertEqual(self.model.output_neurons, 2)

	def test_if_initial_data_is_flattened(self):
		self.assertEqual(self.model.X.shape[0], self.data_length * 2)
		#: monkey patching for data rescale
		self.model.Xf = self.model.X
		self.model._rescale_values()
		rescaled_flat = self.model.Xf
		self.assertEqual(rescaled_flat[0], self.values[0][0])
		self.assertEqual(rescaled_flat[1], self.values[0][1])

	def test_if_training_dataset_is_properly_configured(self):
		trainX = self.model.trainingX
		trainY = self.model.trainingY
		self.assertEqual(len(trainX[0]), self.lookback * 2)
		self.assertEqual(len(trainY), len(self.values) - self.lookback)

	def test_if_can_predict_proper_values(self):
		STEPS = 5
		prediction = self.model.predict(steps=STEPS)
		self.assertEqual(len(prediction), STEPS)
		self.assertEqual(prediction['values'][0].shape, (2,1))