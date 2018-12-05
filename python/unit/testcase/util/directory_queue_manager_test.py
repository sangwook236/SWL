import unittest
from swl.util import directory_queue_manager
import numpy as np

class UtilTestCase(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_getAvailableDirectory(self):
		base_dir_name = 'tmp_dir'
		dir_queue_mgr = directory_queue_manager.DirectoryQueueManager(3, base_dir_name)

		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 0))
		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 1))
		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 2))
		self.assertIsNone(dir_queue_mgr.getAvailableDirectory())

	def test_returnDirectory(self):
		base_dir_name = 'tmp_dir'
		dir_queue_mgr = directory_queue_manager.DirectoryQueueManager(4, base_dir_name)

		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 0))
		self.assertFalse(dir_queue_mgr.returnDirectory('{}_{}'.format(base_dir_name, 1)))
		self.assertFalse(dir_queue_mgr.returnDirectory('{}_{}'.format(base_dir_name, 2)))
		self.assertTrue(dir_queue_mgr.returnDirectory('{}_{}'.format(base_dir_name, 0)))

		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 1))
		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 2))
		self.assertEqual(dir_queue_mgr.getAvailableDirectory(), '{}_{}'.format(base_dir_name, 3))
		self.assertTrue(dir_queue_mgr.returnDirectory('{}_{}'.format(base_dir_name, 3)))
		self.assertTrue(dir_queue_mgr.returnDirectory('{}_{}'.format(base_dir_name, 1)))
