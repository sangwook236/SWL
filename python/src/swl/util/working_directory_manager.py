import os, queue
import swl.util.util as swl_util

class WorkingDirectoryManager(object):
	"""One-step working directory manager.
	"""

	def __init__(self, dir_path_prefix, num_dirs):
		"""
		Inputs:
			dir_path_prefix (string): A path prefix of directories to be managed.
			num_dirs (int): Total number of working directories.
		"""

		super().__init__()

		self._workable_dir_queue = queue.Queue(maxsize=num_dirs)
		self._busy_dir_set = set()

		for idx in range(num_dirs):
			dir_path = '{}_{}'.format(dir_path_prefix, idx)
			swl_util.make_dir(dir_path)
			self._workable_dir_queue.put(dir_path)

	def sizeDirectory(self):
		return self._workable_dir_queue.qsize()

	def requestDirectory(self, block=True, timeout=None):
		if self._workable_dir_queue.empty():
			return None
		else:
			dir_path = self._workable_dir_queue.get(block=block, timeout=timeout)
			self._busy_dir_set.add(dir_path)
			return dir_path

	def returnDirectory(self, dir_path, block=True, timeout=None):
		if dir_path in self._busy_dir_set:
			self._busy_dir_set.remove(dir_path)
			self._workable_dir_queue.put(dir_path, block=block, timeout=timeout)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False

class TwoStepWorkingDirectoryManager(object):
	"""Two-step working directory manager.

	Work step:
		Preparatory directory -> working directory -> preparatory directory -> ...
	"""

	def __init__(self, dir_path_prefix, num_dirs):
		"""
		Inputs:
			dir_path_prefix (string): A path prefix of directories to be managed.
			num_dirs (int): Total number of working directories.
		"""

		super().__init__()

		# Preparatory directories -> for preparation work.
		self._preparatory_dir_queue = queue.Queue(maxsize=num_dirs)
		# Workable directories -> for main work.
		self._workable_dir_queue = queue.Queue(maxsize=num_dirs)
		self._busy_dir_dict = dict()

		for idx in range(num_dirs):
			dir_path = '{}_{}'.format(dir_path_prefix, idx)
			swl_util.make_dir(dir_path)
			self._preparatory_dir_queue.put(dir_path)

	def sizeDirectory(self, is_workable=True):
		return self._workable_dir_queue.qsize() if is_workable else self._preparatory_dir_queue.qsize()

	def requestDirectory(self, is_workable=True, block=True, timeout=None):
		if is_workable:
			if self._workable_dir_queue.empty():
				return None
			else:
				dir_path = self._workable_dir_queue.get(block=block, timeout=timeout)
				self._busy_dir_dict[dir_path] = True
				return dir_path
		else:
			if self._preparatory_dir_queue.empty():
				return None
			else:
				dir_path = self._preparatory_dir_queue.get(block=block, timeout=timeout)
				self._busy_dir_dict[dir_path] = False
				return dir_path

	def returnDirectory(self, dir_path, block=True, timeout=None):
		if dir_path in self._busy_dir_dict:
			if self._busy_dir_dict[dir_path]:  # dir_path was workable.
				self._preparatory_dir_queue.put(dir_path, block=block, timeout=timeout)
			else:  # dir_path was preparatory.
				self._workable_dir_queue.put(dir_path, block=block, timeout=timeout)
			self._busy_dir_dict.pop(dir_path)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False

class MultiStepWorkingDirectoryManager(object):
	"""Multi-step working directory manager.

	Work step:
		Working directory #0 (start working directory) -> working directory #1 -> ... -> working directory #(num_work_steps - 1) (final working directory) -> working directory #0 -> ...
	"""

	def __init__(self, dir_path_prefix, num_dirs, num_work_steps):
		"""
		Inputs:
			dir_path_prefix (string): A path prefix of directories to be managed.
			num_dirs (int): Total number of working directories.
			num_work_steps (int): Number of work steps.
		"""

		super().__init__()

		self._num_work_steps = num_work_steps

		self._workable_dir_queue_dict = dict()
		for dir_id in range(self._num_work_steps):
			self._workable_dir_queue_dict[dir_id] = queue.Queue(maxsize=num_dirs)
		self._busy_dir_dict = dict()

		for idx in range(num_dirs):
			dir_path = '{}_{}'.format(dir_path_prefix, idx)
			swl_util.make_dir(dir_path)
			self._workable_dir_queue_dict[0].put(dir_path)

	def sizeDirectory(self, dir_id=0):
		return self._workable_dir_queue_dict[dir_id].qsize()

	def requestDirectory(self, dir_id=0, block=True, timeout=None):
		if self._workable_dir_queue_dict[dir_id].empty():
			return None
		else:
			dir_path = self._workable_dir_queue_dict[dir_id].get(block=block, timeout=timeout)
			self._busy_dir_dict[dir_path] = dir_id
			return dir_path

	def returnDirectory(self, dir_path, block=True, timeout=None):
		if dir_path in self._busy_dir_dict:
			dir_id = (self._busy_dir_dict[dir_path] + 1) % self._num_work_steps
			self._workable_dir_queue_dict[dir_id].put(dir_path, block=block, timeout=timeout)
			self._busy_dir_dict.pop(dir_path)
			return True
		else:
			#raise ValueError('Invalid directory path: {}'.format(dir_path))
			return False
