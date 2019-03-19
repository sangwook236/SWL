#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time, random
from functools import partial
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import threading
from swl.util.working_directory_manager import WorkingDirectoryManager, TwoStepWorkingDirectoryManager

class LockGuard(object):
	def __init__(self, lock):
		self._lock = lock

	def __enter__(self):
		self._lock.acquire(block=True, timeout=None)
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		self._lock.release()

class WorkingDirectoryGuard(object):
	def __init__(self, dirMgr, lock):
		self._dirMgr = dirMgr
		self._lock = lock
		self._dir_path = None

	@property
	def directory(self):
		return self._dir_path

	def __enter__(self):
		print('\t{}({}): Waiting for a working directory...'.format(os.getpid(), threading.get_ident()))
		while True:
			with self._lock:
			#with LockGuard(self._lock):
				self._dir_path = self._dirMgr.requestDirectory()
			if self._dir_path is not None:
				break
			else:
				time.sleep(0.5)
		print('\t{}({}): Got a working directory: {}.'.format(os.getpid(), threading.get_ident(), self._dir_path))
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		while True:
			is_returned = False
			with self._lock:
			#with LockGuard(self._lock):
				is_returned = self._dirMgr.returnDirectory(self._dir_path)
			if is_returned:
				break
			else:
				time.sleep(0.5)
		print('\t{}({}): Returned a working directory: {}.'.format(os.getpid(), threading.get_ident(), self._dir_path))

class TwoStepWorkingDirectoryGuard(object):
	def __init__(self, dirMgr, is_workable, lock):
		self._dirMgr = dirMgr
		self._is_workable = is_workable
		self._lock = lock
		self._step = 'working' if self._is_workable else 'preparatory'
		self._dir_path = None

	@property
	def directory(self):
		return self._dir_path

	def __enter__(self):
		print('\t{}({}): Waiting for a {} directory...'.format(os.getpid(), threading.get_ident(), self._step))
		while True:
			with self._lock:
			#with LockGuard(self._lock):
				self._dir_path = self._dirMgr.requestDirectory(is_workable=self._is_workable)
			if self._dir_path is not None:
				break
			else:
				time.sleep(0.5)
		print('\t{}({}): Got a {} directory: {}.'.format(os.getpid(), threading.get_ident(), self._step, self._dir_path))
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		while True:
			is_returned = False
			with self._lock:
			#with LockGuard(self._lock):
				is_returned = self._dirMgr.returnDirectory(self._dir_path)
			if is_returned:
				break
			else:
				time.sleep(0.5)
		print('\t{}({}): Returned a {} directory: {}.'.format(os.getpid(), threading.get_ident(), self._step, self._dir_path))

def initialize_working_directory_lock(lock):
	global global_working_directory_lock
	global_working_directory_lock = lock

def worker_thread_proc(dirMgr, num_processes, num_steps, lock):
	print('{}({}): Started a worker thread.'.format(os.getpid(), threading.get_ident()))
	#timeout = 10
	timeout = None
	with mp.Pool(processes=num_processes, initializer=initialize_working_directory_lock, initargs=(lock,)) as pool:
		worker_process_results = pool.map_async(partial(worker_process_proc, dirMgr), [step for step in range(num_steps)])

		worker_process_results.get(timeout)
	print('{}({}): Ended a worker thread.'.format(os.getpid(), threading.get_ident()))

def worker_process_proc(dirMgr, step):
	print('\t{}: Started a worker process: Step #{}.'.format(os.getpid(), step))
	with WorkingDirectoryGuard(dirMgr, global_working_directory_lock) as guard:
		if guard.directory:
			secs = random.randint(1, 5)
			print('\t{}: Do work #{} for {} secs in {}.'.format(os.getpid(), step, secs, guard.directory))
			time.sleep(secs)
			print('\t{}: Did work #{} for {} secs in {}.'.format(os.getpid(), step, secs, guard.directory))
		else:
			raise ValueError('Directory is None')
	print('\t{}: Ended a worker process.'.format(os.getpid()))

def working_directory_manager_test():
	dir_path_prefix = './work_dir'
	num_dirs = 5
	num_processes = 10
	num_steps = 20

	BaseManager.register('WorkingDirectoryManager', WorkingDirectoryManager)
	manager = BaseManager()
	manager.start()

	lock = mp.Lock()
	#lock = mp.Manager().Lock()  # TypeError: can't pickle thread.lock objects.

	dirMgr_mp = manager.WorkingDirectoryManager(dir_path_prefix, num_dirs)

	#--------------------
	worker_thread = threading.Thread(target=worker_thread_proc, args=(dirMgr_mp, num_processes, num_steps, lock))
	worker_thread.start()
	print('{}({}): Started a worker thread.'.format(os.getpid(), threading.get_ident()))

	#--------------------
	worker_thread.join()
	print('{}({}): Joined a worker thread.'.format(os.getpid(), threading.get_ident()))

def initialize_two_step_working_directory_lock(lock):
	global global_two_step_working_directory_lock
	global_two_step_working_directory_lock = lock

def main_worker_thread_proc(dirMgr, num_steps, lock):
	print('{}({}): Started a main worker thread.'.format(os.getpid(), threading.get_ident()))
	for step in range(num_steps):
		with TwoStepWorkingDirectoryGuard(dirMgr, True, lock) as guard:
			if guard.directory:
				secs = random.randint(1, 3)
				print('{}({}): Do work #{} for {} secs in {}.'.format(os.getpid(), threading.get_ident(), step, secs, guard.directory))
				time.sleep(secs)
				print('{}({}): Did work #{} for {} secs in {}.'.format(os.getpid(), threading.get_ident(), step, secs, guard.directory))
			else:
				raise ValueError('Directory is None')
	print('{}({}): Ended a main worker thread.'.format(os.getpid(), threading.get_ident()))

def preparatory_worker_thread_proc(dirMgr, num_processes, num_steps, lock):
	print('{}({}): Started a preparatory worker thread.'.format(os.getpid(), threading.get_ident()))
	#timeout = 10
	timeout = None
	with mp.Pool(processes=num_processes, initializer=initialize_two_step_working_directory_lock, initargs=(lock,)) as pool:
		worker_process_results = pool.map_async(partial(preparatory_worker_process_proc, dirMgr), [step for step in range(num_steps)])

		worker_process_results.get(timeout)
	print('{}({}): Ended a preparatory worker thread.'.format(os.getpid(), threading.get_ident()))

def preparatory_worker_process_proc(dirMgr, step):
	print('\t{}: Started a preparatory worker process: Step #{}.'.format(os.getpid(), step))
	with TwoStepWorkingDirectoryGuard(dirMgr, False, global_two_step_working_directory_lock) as guard:
		if guard.directory:
			secs = random.randint(1, 5)
			print('\t{}: Prepare work #{} for {} secs in {}.'.format(os.getpid(), step, secs, guard.directory))
			time.sleep(secs)
			print('\t{}: Prepared work #{} for {} secs in {}.'.format(os.getpid(), step, secs, guard.directory))
		else:
			raise ValueError('Directory is None')
	print('\t{}: Ended a preparatory worker process.'.format(os.getpid()))

def two_step_working_directory_manager_test():
	dir_path_prefix = './work_dir'
	num_dirs = 5
	num_processes = 10
	num_steps = 20

	BaseManager.register('TwoStepWorkingDirectoryManager', TwoStepWorkingDirectoryManager)
	manager = BaseManager()
	manager.start()

	lock = mp.Lock()
	#lock = mp.Manager().Lock()  # TypeError: can't pickle thread.lock objects.

	dirMgr_mp = manager.TwoStepWorkingDirectoryManager(dir_path_prefix, num_dirs)

	#--------------------
	main_worker_thread = threading.Thread(target=main_worker_thread_proc, args=(dirMgr_mp, num_steps, lock))
	preparatory_worker_thread = threading.Thread(target=preparatory_worker_thread_proc, args=(dirMgr_mp, num_processes, num_steps, lock))
	main_worker_thread.start()
	preparatory_worker_thread.start()
	print('{}({}): Started main and preparatory worker threads.'.format(os.getpid(), threading.get_ident()))

	#--------------------
	main_worker_thread.join()
	preparatory_worker_thread.join()
	print('{}({}): Joined main and preparatory worker threads.'.format(os.getpid(), threading.get_ident()))

def multi_step_working_directory_manager_test():
	raise NotImplementedError

def main():
	working_directory_manager_test()
	#two_step_working_directory_manager_test()
	#multi_step_working_directory_manager_test()  # Not yet implemented.

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
