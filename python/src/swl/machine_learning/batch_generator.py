import os, abc, csv
import numpy as np

#%%------------------------------------------------------------------
# BatchGenerator.
#	Generates batches.
class BatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a generator.
	@abc.abstractmethod
	def generateBatches(self, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# FileBatchGenerator.
#	Generates batches and saves them to files.
class FileBatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def saveBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# SimpleBatchGenerator.
#	Generates batches from numpy.array.
class SimpleBatchGenerator(BatchGenerator):
	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, input_filepaths=None, output_filepaths=None):
		"""
		Inputs:
			inputs (numpy.array): Input data of type numpy.array. It can be None.
			outputs (numpy.array): Output data of type numpy.array. It can be None.
			input_filepaths (a list of strings): A list of input npy files.
			output_filepaths (a list of strings): A list of output npy files.
			augmenter (object):
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
		"""

		super().__init__()

		batch_axis = 1 if is_time_major else 0
		self._inputs, self._outputs = inputs, outputs
		if input_filepaths is not None and output_filepaths is not None:
			if len(input_filepaths) != len(output_filepaths):
				raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
			for image_filepath, label_filepath in zip(input_filepaths, output_filepaths):
				inp = np.load(image_filepath)
				outp = np.load(label_filepath)
				if inp.shape[batch_axis] != outp.shape[batch_axis]:
					raise ValueError('Unmatched shapes of {} and {}'.format(image_filepath, label_filepath))
				self._inputs = inp if self._inputs is None else np.concatenate((self._inputs, inp), axis=0)
				self._outputs = outp if self._outputs is None else np.concatenate((self._outputs, outp), axis=0)
		if self._inputs is None or self._outputs is None or self._inputs.shape[batch_axis] != self._outputs.shape[batch_axis]:
			raise ValueError('Invalid inputs or outputs')

		self._batch_size = batch_size
		self._shuffle = shuffle
		self._augmenter = augmenter
		self._is_output_augmented = is_output_augmented

		self._num_examples = self._inputs.shape[batch_axis]
		self._num_steps = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._inputs is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid number of examples')

	def generateBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				batch_inputs, batch_outputs = self._inputs[batch_indices], self._outputs[batch_indices]
				if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
					if self._augmenter is None:
						yield batch_inputs, batch_outputs
					else:
						yield self._augmenter(batch_inputs, batch_outputs, self._is_output_augmented)

#%%------------------------------------------------------------------
# NpyFileBatchGenerator.
#	Generates batches from numpy.array and saves them to npy files.
class NpyFileBatchGenerator(FileBatchGenerator):
	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None, input_filepaths=None, output_filepaths=None):
		"""
		Inputs:
			inputs (numpy.array): Input data of type numpy.array. It can be None.
			outputs (numpy.array): Output data of type numpy.array. It can be None.
			input_filepaths (a list of strings): A list of input npy files.
			output_filepaths (a list of strings): A list of output npy files.
				In this constructor, all data will be loaded from input and output npy files.
			augmenter (object):
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
		"""

		super().__init__()

		batch_axis = 1 if is_time_major else 0
		self._inputs, self._outputs = inputs, outputs
		if input_filepaths is not None and output_filepaths is not None:
			if len(input_filepaths) != len(output_filepaths):
				raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
			for image_filepath, label_filepath in zip(input_filepaths, output_filepaths):
				inp = np.load(image_filepath)
				outp = np.load(label_filepath)
				if inp.shape[batch_axis] != outp.shape[batch_axis]:
					raise ValueError('Unmatched shapes of {} and {}'.format(image_filepath, label_filepath))
				self._inputs = inp if self._inputs is None else np.concatenate((self._inputs, inp), axis=0)
				self._outputs = outp if self._outputs is None else np.concatenate((self._outputs, outp), axis=0)
		if self._inputs is None or self._outputs is None or self._inputs.shape[batch_axis] != self._outputs.shape[batch_axis]:
			raise ValueError('Invalid inputs or outputs')

		self._batch_size = batch_size
		self._shuffle = shuffle
		self._augmenter = augmenter
		self._is_output_augmented = is_output_augmented

		self._batch_input_filename_format = 'batch_input_{}.npy' if batch_input_filename_format is None else batch_input_filename_format
		self._batch_output_filename_format = 'batch_output_{}.npy' if batch_output_filename_format is None else batch_output_filename_format
		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

		self._num_examples = self._inputs.shape[batch_axis]
		self._num_steps = ((self._num_examples - 1) // self._batch_size + 1) if self._num_examples > 0 else 0
		#if self._inputs is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid number of examples')

	def saveBatches(self, dir_path, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		with open(os.path.join(dir_path, self._batch_info_csv_filename), 'w', encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)

			for step in range(self._num_steps):
				start = step * self._batch_size
				end = start + self._batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					batch_inputs, batch_outputs = self._inputs[batch_indices], self._outputs[batch_indices]
					if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
						if self._augmenter is not None:
							batch_inputs, batch_outputs = self._augmenter(batch_inputs, batch_outputs, self._is_output_augmented)
						input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename_format.format(step)), os.path.join(dir_path, self._batch_output_filename_format.format(step))
						np.save(input_filepath, batch_inputs)
						np.save(output_filepath, batch_outputs)
						writer.writerow((input_filepath, output_filepath, len(batch_indices)))

#%%------------------------------------------------------------------
# NpyFileBatchGeneratorWithFileInput.
#	Loads data from npy files, generates their batches and saves them to npy files.
class NpyFileBatchGeneratorWithFileInput(FileBatchGenerator):
	def __init__(self, input_filepaths, output_filepaths, num_loaded_files, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None):
		"""
		Inputs:
			input_filepaths (a list of strings): A list of input npy files.
			output_filepaths (a list of strings): A list of output npy files.
				In this constructor, any data will not be loaded from input and output npy files.
			num_loaded_files (int): The number of files that can be loaded at a time.
			augmenter (object):
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
		"""

		super().__init__()

		if num_loaded_files <= 0:
			raise ValueError('Invalid number of files that can be loaded at one time')

		batch_axis = 1 if is_time_major else 0
		if input_filepaths is None or output_filepaths is None:
			raise ValueError('input_filepaths or output_filepaths will not be None')
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
		for image_filepath, label_filepath in zip(input_filepaths, output_filepaths):
			inp = np.load(image_filepath)
			outp = np.load(label_filepath)
			if inp.shape[batch_axis] != outp.shape[batch_axis]:
				raise ValueError('Unmatched shapes of {} and {}'.format(image_filepath, label_filepath))
		self._input_filepaths, self._output_filepaths = input_filepaths, output_filepaths
		self._num_loaded_files = num_loaded_files
		self._num_files = len(self._input_filepaths)
		self._num_file_groups = ((self._num_files - 1) // self._num_loaded_files + 1) if self._num_files > 0 else 0
		if self._num_file_groups <= 0:
			raise ValueError('Invalid number of file groups')

		self._batch_size = batch_size
		self._shuffle = shuffle
		self._batch_axis = batch_axis
		self._augmenter = augmenter
		self._is_output_augmented = is_output_augmented

		self._batch_input_filename_format = 'batch_input_{}.npy' if batch_input_filename_format is None else batch_input_filename_format
		self._batch_output_filename_format = 'batch_output_{}.npy' if batch_output_filename_format is None else batch_output_filename_format
		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

	def saveBatches(self, dir_path, *args, **kwargs):
		file_indices = np.arange(self._num_files)
		if self._shuffle:
			np.random.shuffle(file_indices)

		start_file_index = 0
		for gid in range(self._num_file_groups):
			start = gid * self._num_loaded_files
			end = start + self._num_loaded_files
			sub_file_indices = file_indices[start:end]
			if sub_file_indices.size > 0:  # If sub_file_indices is non-empty.
				sub_input_filepaths = self._input_filepaths[sub_file_indices]
				sub_output_filepaths = self._output_filepaths[sub_file_indices]
				if sub_input_filepaths.size > 0 and sub_output_filepaths.size > 0:  # If sub_input_filepaths and sub_output_filepaths are non-empty.
					inputs, outputs = NpyFileBatchGeneratorWithFileInput._load_data(sub_input_filepaths, sub_output_filepaths, self._batch_axis)
					num_generated_files = self._save_batches(dir_path, inputs, outputs, start_file_index, 'w' if 0 == gid else 'a')
					start_file_index += num_generated_files

	def _save_batches(self, dir_path, inputs, outputs, start_file_index, mode):
		num_examples = inputs.shape[self._batch_axis]
		if num_examples <= 0:
			raise ValueError('Invalid number of examples')
		num_steps = ((num_examples - 1) // self._batch_size + 1) if num_examples > 0 else 0
		if num_steps <= 0:
			raise ValueError('Invalid number of steps')

		indices = np.arange(num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		with open(os.path.join(dir_path, self._batch_info_csv_filename), mode=mode, encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)

			for step in range(num_steps):
				start = step * self._batch_size
				end = start + self._batch_size
				batch_indices = indices[start:end]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					batch_inputs, batch_outputs = inputs[batch_indices], outputs[batch_indices]
					if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
						if self._augmenter is not None:
							batch_inputs, batch_outputs = self._augmenter(batch_inputs, batch_outputs, self._is_output_augmented)
						input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename_format.format(start_file_index + step)), os.path.join(dir_path, self._batch_output_filename_format.format(start_file_index + step))
						np.save(input_filepath, batch_inputs)
						np.save(output_filepath, batch_outputs)
						writer.writerow((input_filepath, output_filepath, len(batch_indices)))

		return num_steps

	@staticmethod
	def _load_data(input_filepaths, output_filepaths, batch_axis):
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
		inputs, outputs = None, None
		for image_filepath, label_filepath in zip(input_filepaths, output_filepaths):
			inp = np.load(image_filepath)
			outp = np.load(label_filepath)
			if inp.shape[batch_axis] != outp.shape[batch_axis]:
				raise ValueError('Unmatched shapes of {} and {}'.format(image_filepath, label_filepath))
			inputs = inp if inputs is None else np.concatenate((inputs, inp), axis=0)
			outputs = outp if outputs is None else np.concatenate((outputs, outp), axis=0)

		return inputs, outputs
