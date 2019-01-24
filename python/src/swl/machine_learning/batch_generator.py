import os, abc, csv
import numpy as np

#%%------------------------------------------------------------------
# BatchGenerator.
#	Generates batches.
class BatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a batch generator.
	@abc.abstractmethod
	def generateBatches(self, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# FileBatchGenerator.
#	Generates and saves batches to files.
class FileBatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a list of filepath pairs, (input filepath, output filepath).
	@abc.abstractmethod
	def saveBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# FileBatchLoader.
#	Loads batches from files.
class FileBatchLoader(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a batch generator.
	@abc.abstractmethod
	def loadBatches(self, filepath_pairs, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# SimpleBatchGenerator.
#	Generates batches.
class SimpleBatchGenerator(BatchGenerator):
	# functor: inputs, outputs = functor(inputs, outputs).
	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, functor=None):
		super().__init__()

		self._inputs = inputs
		self._outputs = outputs
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._functor = functor

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._inputs is not None:
			self._num_examples = self._inputs.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._inputs is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def generateBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_inputs = self._inputs[batch_indices]
				batch_outputs = self._outputs[batch_indices]
				if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
					if self._functor is None:
						yield batch_inputs, batch_outputs
					else:
						yield self._functor(batch_inputs, batch_outputs)

#%%------------------------------------------------------------------
# NpyFileBatchGenerator.
#	Generates and saves batches to npy files.
class NpyFileBatchGenerator(FileBatchGenerator):
	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, functor=None, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None):
		super().__init__()

		self._inputs = inputs
		self._outputs = outputs
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._functor = functor

		self._batch_input_filename_format = 'batch_input_{}.npy' if batch_input_filename_format is None else batch_input_filename_format
		self._batch_output_filename_format = 'batch_output_{}.npy' if batch_output_filename_format is None else batch_output_filename_format
		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._inputs is not None:
			self._num_examples = self._inputs.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // self._batch_size + 1) if self._num_examples > 0 else 0
		#if self._inputs is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def saveBatches(self, dir_path, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		with open(os.path.join(dir_path, self._batch_info_csv_filename ), 'w', encoding='UTF8') as csvfile:
			writer = csv.writer(csvfile)

			for step in range(self._num_steps):
				start = step * self._batch_size
				end = start + self._batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					batch_inputs = self._inputs[batch_indices]
					batch_outputs = self._outputs[batch_indices]
					if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
						if self._functor is not None:
							batch_inputs, batch_outputs = self._functor(batch_inputs, batch_outputs)
						input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename_format.format(step)), os.path.join(dir_path, self._batch_output_filename_format.format(step))
						np.save(input_filepath, batch_inputs)
						np.save(output_filepath, batch_outputs)
						writer.writerow((input_filepath, output_filepath))

#%%------------------------------------------------------------------
# NpyFileBatchLoader.
#	Loads batches from npy files.
class NpyFileBatchLoader(FileBatchLoader):
	def __init__(self, batch_info_csv_filename=None):
		super().__init__()

		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

	def loadBatches(self, dir_path, *args, **kwargs):
		with open(os.path.join(dir_path, self._batch_info_csv_filename), 'r', encoding='UTF8') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if not row:
					continue
				batch_inputs = np.load(row[0])
				batch_outputs = np.load(row[1])
				yield batch_inputs, batch_outputs
