import os, abc, csv, zipfile
import numpy as np

#--------------------------------------------------------------------

class FileBatchLoader(abc.ABC):
	"""Loads batches from files.
	"""

	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def loadBatches(self, dir_path, *args, **kwargs):
		"""Returns a generator.

		Inputs:
			dir_path (string): A path of a directory to load batch files.
		Outputs:
			A generator to generate batches.
		"""

		raise NotImplementedError

	@abc.abstractmethod
	def loadBatchesUsingDirectoryGuard(self, directoryGuard, *args, **kwargs):
		"""Returns a generator.

		Inputs:
			directoryGuard (object): A guard of a directory to load batch files.
				The guard has a property 'directory' which returns a directory to load batch files.
				The guard has attributes __enter__() and __exit__() for the with statement.
		Outputs:
			A generator to generate batches.
		"""

		raise NotImplementedError

#--------------------------------------------------------------------

class NpzFileBatchLoader(FileBatchLoader):
	"""Loads batches from npz files.
	"""

	def __init__(self, batch_info_csv_filename=None, data_processing_functor=None):
		"""
		Inputs:
			batch_info_csv_filename (string): A CSV file name to save info about batch inputs and outputs. batch_info_csv_filename = 'batch_info.csv' when it is None.
			data_processing_functor (functor): A function object for data processing. It can be None.
		"""

		super().__init__()

		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename
		self._data_processing_functor = data_processing_functor

	def loadBatches(self, dir_path, *args, **kwargs):
		"""Returns a generator.

		Inputs:
			dir_path (string): The path of a directory to load batch files.
		Outputs:
			A generator to generate batches.
		"""

		with open(os.path.join(dir_path, self._batch_info_csv_filename), 'r', encoding='UTF8') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if not row:
					continue
				try:
					batch_inputs_npzfile = np.load(row[0])
					batch_outputs_npzfile = np.load(row[1])
				except (IOError, ValueError) as ex:
					continue
				num_all_examples = int(row[2])

				#for ki, ko in zip(sorted(batch_inputs_npzfile.keys()), sorted(batch_outputs_npzfile.keys())):
				for ki, ko in zip(batch_inputs_npzfile.keys(), batch_outputs_npzfile.keys()):
					if ki != ko:
						print('Unmatched batch key name: {} != {}.'.format(ki, ko))
						continue

					try:
						batch_inputs, batch_outputs = batch_inputs_npzfile[ki], batch_outputs_npzfile[ko]
					except zipfile.BadZipFile as ex:
						print('Zip file error: {} in {} or {}: {}.'.format(ki, row[0], row[1], ex))
						continue
					"""
					try:
						batch_inputs = batch_inputs_npzfile[ki]
					except zipfile.BadZipFile as ex:
						print('Zip file error: {} in {}: {}.'.format(ki, row[0], ex))
						continue
					try:
						batch_outputs = batch_outputs_npzfile[ko]
					except zipfile.BadZipFile as ex:
						print('Zip file error: {} in {}: {}.'.format(ko, row[1], ex))
						continue
					"""

					num_batch_examples = len(batch_inputs)
					if num_batch_examples != len(batch_outputs):
						print('Unmatched batch size: {} != {}.'.format(num_batch_examples, len(batch_outputs)))
						continue
					if self._data_processing_functor is not None:
						batch_inputs, batch_outputs = self._data_processing_functor(batch_inputs, batch_outputs)
					yield (batch_inputs, batch_outputs), num_batch_examples

	def loadBatchesUsingDirectoryGuard(self, directoryGuard, *args, **kwargs):
		"""Returns a generator.

		Inputs:
			directoryGuard (object): A guard of a directory to load batch files.
				The guard has a property 'directory' which returns a directory to load batch files.
				The guard has attributes __enter__() and __exit__() for the with statement.
		Outputs:
			A generator to generate batches.
		"""

		with directoryGuard:
			if directoryGuard.directory is None:
				raise ValueError('Directory is None')

			"""
			# NOTE [info] >> This implementation does not properly work because of the characteristic of yield keyword.
			return self.loadBatches(directoryGuard.directory, *args, **kwargs)
			"""
			with open(os.path.join(directoryGuard.directory, self._batch_info_csv_filename), 'r', encoding='UTF8') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					if not row:
						continue
					try:
						batch_inputs_npzfile = np.load(row[0])
						batch_outputs_npzfile = np.load(row[1])
					except (IOError, ValueError) as ex:
						continue
					num_all_examples = int(row[2])

					#for ki, ko in zip(sorted(batch_inputs_npzfile.keys()), sorted(batch_outputs_npzfile.keys())):
					for ki, ko in zip(batch_inputs_npzfile.keys(), batch_outputs_npzfile.keys()):
						if ki != ko:
							print('Unmatched batch key name: {} != {}.'.format(ki, ko))
							continue

						try:
							batch_inputs, batch_outputs = batch_inputs_npzfile[ki], batch_outputs_npzfile[ko]
						except zipfile.BadZipFile as ex:
							print('Zip file error: {} in {} or {}: {}.'.format(ki, row[0], row[1], ex))
							continue
						"""
						try:
							batch_inputs = batch_inputs_npzfile[ki]
						except zipfile.BadZipFile as ex:
							print('Zip file error: {} in {}: {}.'.format(ki, row[0], ex))
							continue
						try:
							batch_outputs = batch_outputs_npzfile[ko]
						except zipfile.BadZipFile as ex:
							print('Zip file error: {} in {}: {}.'.format(ko, row[1], ex))
							continue
						"""

						num_batch_examples = len(batch_inputs)
						if num_batch_examples != len(batch_outputs):
							print('Unmatched batch size: {} != {}.'.format(num_batch_examples, len(batch_outputs)))
							continue
						if self._data_processing_functor is not None:
							batch_inputs, batch_outputs = self._data_processing_functor(batch_inputs, batch_outputs)
						yield (batch_inputs, batch_outputs), num_batch_examples
