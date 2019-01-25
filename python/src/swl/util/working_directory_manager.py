import os, queue
import swl.util.util as swl_util

class SimpleWorkingDirectoryManager(object):
	def __init__(self, dir_path_prefix, num_dirs):
		super().__init__()

		self._available_dir_queue = queue.Queue(maxsize=num_dirs)
		self._busy_dir_set = set()

		for idx in range(num_dirs):
			dir_path = '{}_{}'.format(dir_path_prefix, idx)
			swl_util.make_dir(dir_path)
			self._available_dir_queue.put(dir_path)

	def sizeDirectory(self):
		return self._available_dir_queue.qsize()

	def requestDirectory(self, block=True, timeout=None):
		if self._available_dir_queue.empty():
			return None
		else:
			dir_path = self._available_dir_queue.get(block=block, timeout=timeout)
			self._busy_dir_set.add(dir_path)
			return dir_path

	def returnDirectory(self, dir_path, block=True, timeout=None):
		if dir_path in self._busy_dir_set:
			self._busy_dir_set.remove(dir_path)
			self._available_dir_queue.put(dir_path, block=block, timeout=timeout)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False

class ReadyWorkingDirectoryManager(object):
	def __init__(self, dir_path_prefix, num_dirs):
		super().__init__()

		# Available directories -> for preparation work.
		self._available_dir_queue = queue.Queue(maxsize=num_dirs)
		# Ready directories -> for main work.
		self._ready_dir_queue = queue.Queue(maxsize=num_dirs)
		self._busy_dir_dict = dict()

		for idx in range(num_dirs):
			dir_path = '{}_{}'.format(dir_path_prefix, idx)
			swl_util.make_dir(dir_path)
			self._available_dir_queue.put(dir_path)

	def sizeDirectory(self):
		return self._available_dir_queue.qsize()

	def sizeReadyDirectory(self):
		return self._ready_dir_queue.qsize()

	def requestDirectory(self, block=True, timeout=None):
		if self._available_dir_queue.empty():
			return None
		else:
			dir_path = self._available_dir_queue.get(block=block, timeout=timeout)
			self._busy_dir_dict[dir_path] = False
			return dir_path

	def requestReadyDirectory(self, block=True, timeout=None):
		if self._ready_dir_queue.empty():
			return None
		else:
			dir_path = self._ready_dir_queue.get(block=block, timeout=timeout)
			self._busy_dir_dict[dir_path] = True
			return dir_path

	def returnDirectory(self, dir_path, block=True, timeout=None):
		if dir_path in self._busy_dir_dict:
			if self._busy_dir_dict[dir_path]:  # dir_path was ready.
				self._available_dir_queue.put(dir_path, block=block, timeout=timeout)
			else:  # dir_path was available.
				self._ready_dir_queue.put(dir_path, block=block, timeout=timeout)
			self._busy_dir_dict.pop(dir_path)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False
