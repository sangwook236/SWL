import os, abc
import queue
import swl.util.util as swl_util

class DirectoryQueueManager(abc.ABC):
	def __init__(self, maxsize, base_dir_path):
		super().__init__()

		self._available_dir_queue = queue.Queue(maxsize=maxsize)
		self._busy_dir_set = set()

		for idx in range(maxsize):
			dir_path = '{}_{}'.format(base_dir_path, idx)
			swl_util.make_dir(dir_path)
			self._available_dir_queue.put(dir_path)

	def getAvailableDirectory(self):
		if self._available_dir_queue.empty():
			return None
		else:
			dir_path = self._available_dir_queue.get()
			self._busy_dir_set.add(dir_path)
			return dir_path

	def returnDirectory(self, dir_path):
		if dir_path in self._busy_dir_set:
			self._busy_dir_set.remove(dir_path)
			self._available_dir_queue.put(dir_path)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False
