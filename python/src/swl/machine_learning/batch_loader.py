import os, abc, csv, zipfile
import numpy as np

#%%------------------------------------------------------------------
# FileBatchLoader.
#	Loads batches from files.
class FileBatchLoader(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a generator.
	@abc.abstractmethod
	def loadBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

	# Returns a generator.
	@abc.abstractmethod
	def loadBatchesUsingDirectoryGuard(self, directoryGuard, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# NpzFileBatchLoader.
#	Loads batches from npz files.
class NpzFileBatchLoader(FileBatchLoader):
	def __init__(self, batch_info_csv_filename=None, data_processing_functor=None):
		super().__init__()

		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename
		self._data_processing_functor = data_processing_functor

	# Returns a generator.
	def loadBatches(self, dir_path, *args, **kwargs):
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

	# Returns a generator.
	def loadBatchesUsingDirectoryGuard(self, directoryGuard, *args, **kwargs):
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
