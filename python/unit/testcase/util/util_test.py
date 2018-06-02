import unittest
from swl.util import util
import numpy as np

class UtilTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_add(self):
		arr = np.array([1, 2, 2, 3, 4, 5, 4, 5, 3, 5, 5, 2])
		self.assertEqual(util.find_most_frequent_value(arr), 5)
		arr = np.array([1, 2, 2, 3, 4, 5, 4, 5, 3, 5, 5, 2, 2, 0, 2])
		self.assertEqual(util.find_most_frequent_value(arr), 2)
