#: -*- coding: utf-8 -*-
"""
Time series prediction Artificial Neural Network model tests
"""
import unittest
import pandas as pd

from ipredictor.models import ANN


WEIGHTS_FILE = 'assets/weights.h5'


@unittest.skip("")
class ANNTestCase(unittest.TestCase):

	def setUp(self):
		self.lookback = 2
		self.values = range(1, self.lookback * 4 + 1)
		self.dataframe = pd.DataFrame.from_items([('values', self.values)])
		self.model = ANN(self.dataframe, lookback=self.lookback)

	def test_if_initial_user_data_is_scaled(self):
		self.assertTrue(0 <= self.model.X[-1] <= 1)

	def test_if_result_values_can_be_rescaled_back(self):
		self.model.Xf = self.model.X
		self.model._rescale_values()
		self.assertEqual(self.model.Xf[-1], self.values[-1])

	def test_if_model_validates_and_uses_predefined_coefs(self):
		bad = 'badfile.txt'
		self.assertRaises(ValueError, self.model._check_initial_coefs, bad)

		good = WEIGHTS_FILE
		self.model.predict()
		self.model.save_coefs(good)
		try:
			self.model._check_initial_coefs(good)
		except ValueError:
			self.fail("Unexpected error raised")

		self.model.coefs = WEIGHTS_FILE
		self.model.predict()
		self.assertIsNotNone(self.model.coefs)
		condition = self.model.coefs is not True
		self.assertTrue(condition)

	def test_if_hidden_and_input_neurons_count_properly_calculated(self):
		self.assertEqual(self.model.input_neurons, self.lookback)
		self.assertEqual(self.model.hidden_neurons, self.lookback * 4)

	def test_if_trainig_data_generated_properly(self):
		trainX = self.model.trainingX
		trainY = self.model.trainingY
		expected_len = len(self.values) - self.lookback
		self.assertEqual(len(trainX), expected_len)
		self.assertEqual(len(trainY), expected_len)
		self.assertEqual(len(trainX[0]), self.lookback)
		self.assertEqual(self.model.X[self.lookback], trainY[0])

	def	test_if_automatically_finds_model_weights_if_not_provided(self):
		self.model.predict()
		self.assertIsNotNone(self.model.coefs)

	def test_if_predicts_future_values_properly(self):
		STEPS = 10
		prediction = self.model.predict(steps=STEPS)
		self.assertEqual(len(prediction), STEPS)
		#: if data rescaled back
		self.assertTrue(any([1 < x or x < 0 for x in prediction['values']]))

