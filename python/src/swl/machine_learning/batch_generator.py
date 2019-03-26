import os, abc, math, csv
import numpy as np
import cv2 as cv

#--------------------------------------------------------------------

class BatchGenerator(abc.ABC):
	"""Generates batches.
	"""

	def __init__(self):
		super().__init__()

	# Returns a generator.
	@abc.abstractmethod
	def generateBatches(self, *args, **kwargs):
		raise NotImplementedError

#--------------------------------------------------------------------

class FileBatchGenerator(abc.ABC):
	"""Generates batches and saves them to files.
	"""

	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def saveBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

#--------------------------------------------------------------------

class SimpleBatchGenerator(BatchGenerator):
	"""Generates batches from numpy.array.
	"""

	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, input_filepaths=None, output_filepaths=None):
		"""
		Inputs:
			inputs (numpy.array): Input data of type numpy.array. It can be None.
			outputs (numpy.array): Output data of type numpy.array. It can be None.
			batch_size (int): Batch size.
			shuffle (bool): Specifies whether data is shuffled or not.
			is_time_major (bool): Specifies whether inputs is time-major or batch-major.
			augmenter (object): Data augmenter. It can be None.
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
			is_output_augmented (bool): Specifies whether outputs is augmented or not.
			input_filepaths (a list of strings): A list of input npy files. It can be None.
			output_filepaths (a list of strings): A list of output npy files. It can be None.
		"""

		super().__init__()

		batch_axis = 1 if is_time_major else 0
		self._inputs, self._outputs = inputs, outputs
		if input_filepaths is not None and output_filepaths is not None:
			if len(input_filepaths) != len(output_filepaths):
				raise ValueError('Unmatched lengths of input and output filepaths')
			for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
				inp = np.load(input_filepath)
				outp = np.load(output_filepath)
				if inp.shape[batch_axis] != outp.shape[batch_axis]:
					raise ValueError('Unmatched shapes of {} and {}'.format(input_filepath, output_filepath))
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

	# Returns a generator.
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
						yield (batch_inputs, batch_outputs), batch_indices.size
					else:
						yield self._augmenter(batch_inputs, batch_outputs, self._is_output_augmented), batch_indices.size

#--------------------------------------------------------------------

class NpzFileBatchGenerator(FileBatchGenerator):
	"""Generates batches from numpy.array and saves them to npz files.
	"""

	def __init__(self, inputs, outputs, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename=None, batch_output_filename=None, batch_info_csv_filename=None, input_filepaths=None, output_filepaths=None):
		"""
		Inputs:
			inputs (numpy.array): Input data of type numpy.array. It can be None.
			outputs (numpy.array): Output data of type numpy.array. It can be None.
			batch_size (int): Batch size.
			shuffle (bool): Specifies whether data is shuffled or not.
			is_time_major (bool): Specifies whether inputs is time-major or batch-major.
			augmenter (object): Data augmenter. It can be None.
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
			is_output_augmented (bool): Specifies whether outputs is augmented or not.
			batch_input_filename (string): A file name to save batch inputs. batch_input_filename = 'batch_input.npz' when it is None.
			batch_output_filename (string): A file name to save batch outputs. batch_output_filename = 'batch_output.npz' when it is None.
			batch_info_csv_filename (string): A CSV file name to save info about batch inputs and outputs. batch_info_csv_filename = 'batch_info.csv' when it is None.
			input_filepaths (a list of strings): A list of input npy files. It can be None.
			output_filepaths (a list of strings): A list of output npy files. It can be None.
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

		self._batch_input_filename = 'batch_input.npz' if batch_input_filename is None else batch_input_filename
		self._batch_output_filename = 'batch_output.npz' if batch_output_filename is None else batch_output_filename
		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

		self._num_examples = self._inputs.shape[batch_axis]
		self._num_steps = ((self._num_examples - 1) // self._batch_size + 1) if self._num_examples > 0 else 0
		#if self._inputs is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid number of examples')

	def saveBatches(self, dir_path, *args, **kwargs):
		"""Saves batches.

		Inputs:
			dir_path (string): A directory path to save batches.
		Outputs:
			The number of saved examples.
		"""

		if self._augmenter is not None and self._augmenter._augmenter is not None and isinstance(self._augmenter._augmenter, iaa.Sequential):
			return self._saveBatchesByImgaug(dir_path, *args, **kwargs)
		else:
			return self._saveBatches(dir_path, *args, **kwargs)

	def _saveBatches(self, dir_path, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		"""
		batch_inputs_dict, batch_outputs_dict = dict(), dict()
		num_saved_examples = 0
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
					batch_name = 'batch_{}'.format(step)
					batch_inputs_dict[batch_name], batch_outputs_dict[batch_name] = batch_inputs, batch_outputs
					num_saved_examples += len(batch_indices)
		"""
		if self._augmenter is None:
			inputs, outputs = self._inputs, self._outputs
		else:
			inputs, outputs = self._augmenter(self._inputs, self._outputs, self._is_output_augmented)

		batch_inputs_dict, batch_outputs_dict = dict(), dict()
		num_saved_examples = 0
		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				batch_inputs, batch_outputs = inputs[batch_indices], outputs[batch_indices]
				if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
					batch_name = 'batch_{}'.format(step)
					batch_inputs_dict[batch_name], batch_outputs_dict[batch_name] = batch_inputs, batch_outputs
					num_saved_examples += len(batch_indices)

		#--------------------
		input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename), os.path.join(dir_path, self._batch_output_filename)
		np.savez(input_filepath, **batch_inputs_dict)
		np.savez(output_filepath, **batch_outputs_dict)

		with open(os.path.join(dir_path, self._batch_info_csv_filename), 'w', encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow((input_filepath, output_filepath, num_saved_examples))

		return num_saved_examples

	def _saveBatchesByImgaug(self, dir_path, *args, **kwargs):
		"""Saves batches in parallel by imgaug library.

		Inputs:
			dir_path (string): A directory path to save batches.
		Outputs:
			The number of saved examples.
		"""

		import imgaug_util

		batch_generator = imgaug_util.generateBatchesInParallelWithOutputAugmentation if self._is_output_augmented else imgaug_util.generateBatchesInParallelWithoutOutputAugmentation

		processes, chunksize = 4, 5
		batch_inputs_dict, batch_outputs_dict = dict(), dict()
		num_saved_examples = 0
		# No data preprocessing in batch_generator.
		#	Data preprocessing is performed in NpzFileBatchLoader.
		for idx, (batch_data, num_batch_examples) in enumerate(batch_generator(processes, chunksize, self._augmenter._augmenter, None, self._inputs, self._outputs, self._batch_size, self._shuffle)):
			batch_name = 'batch_{}'.format(idx)
			batch_inputs_dict[batch_name], batch_outputs_dict[batch_name] = batch_data[:2]
			num_saved_examples += num_batch_examples

		#--------------------
		input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename), os.path.join(dir_path, self._batch_output_filename)
		np.savez(input_filepath, **batch_inputs_dict)
		np.savez(output_filepath, **batch_outputs_dict)

		with open(os.path.join(dir_path, self._batch_info_csv_filename), 'w', encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow((input_filepath, output_filepath, num_saved_examples))

		return num_saved_examples

#--------------------------------------------------------------------

class NpzFileBatchGeneratorFromFiles(FileBatchGenerator):
	"""Loads data from files, generates their batches and saves them to npz files.
	"""

	def __init__(self, num_loaded_files, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None):
		"""
		Inputs:
			num_loaded_files (int): The number of files that can be loaded at a time.
			batch_size (int): Batch size.
			shuffle (bool): Specifies whether data is shuffled or not.
			is_time_major (bool): Specifies whether inputs is time-major or batch-major.
			augmenter (object): Data augmenter. It can be None.
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
			is_output_augmented (bool): Specifies whether outputs is augmented or not.
			batch_input_filename_format (string): A file name format to save batch inputs. batch_input_filename_format = 'batch_input_{}.npz' when it is None.
			batch_output_filename_format (string): A file name format to save batch outputs. batch_output_filename_format = 'batch_output_{}.npz' when it is None.
			batch_info_csv_filename (string): A CSV file name to save info about batch inputs and outputs. batch_info_csv_filename = 'batch_info.csv' when it is None.
		"""

		if num_loaded_files <= 0:
			raise ValueError('Invalid number of files that can be loaded at one time')

		self._num_loaded_files = num_loaded_files
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._batch_axis = 1 if is_time_major else 0
		self._augmenter = augmenter
		self._is_output_augmented = is_output_augmented

		self._batch_input_filename_format = 'batch_input_{}.npz' if batch_input_filename_format is None else batch_input_filename_format
		self._batch_output_filename_format = 'batch_output_{}.npz' if batch_output_filename_format is None else batch_output_filename_format
		self._batch_info_csv_filename = 'batch_info.csv' if batch_info_csv_filename is None else batch_info_csv_filename

	def _saveBatchesToNpzFiles(self, inputs, outputs, dir_path, num_files, start_batch_file_idx=0):
		num_examples = inputs.shape[self._batch_axis]
		if num_examples <= 0:
			raise ValueError('Invalid number of examples')

		example_indices = np.arange(num_examples)
		if self._shuffle:
			np.random.shuffle(example_indices)

		# TODO [enhance] >> Augment in parallel by imgaug library.
		if self._augmenter is not None:
			inputs, outputs = self._augmenter(inputs, outputs, self._is_output_augmented)

		num_examples_in_a_file = math.ceil(num_examples / num_files)
		total_num_saved_example, batch_file_idx = 0, 0
		saved_file_info = list()
		for idx in range(num_files):
			sub_example_indices = example_indices[(num_examples_in_a_file * idx):(num_examples_in_a_file * (idx + 1))]
			if sub_example_indices.size > 0:  # If sub_example_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				sub_inputs, sub_outputs = inputs[sub_example_indices], outputs[sub_example_indices]

				batch_inputs_dict, batch_outputs_dict, num_saved_examples = self._constructBatchDictionaries(sub_inputs, sub_outputs)

				input_filepath, output_filepath = os.path.join(dir_path, self._batch_input_filename_format.format(start_batch_file_idx + batch_file_idx)), os.path.join(dir_path, self._batch_output_filename_format.format(start_batch_file_idx + batch_file_idx))
				np.savez(input_filepath, **batch_inputs_dict)
				np.savez(output_filepath, **batch_outputs_dict)

				saved_file_info.append((input_filepath, output_filepath, num_saved_examples))

				total_num_saved_example += num_saved_examples
				batch_file_idx += 1

		return saved_file_info, total_num_saved_example

	def _constructBatchDictionaries(self, inputs, outputs):
		"""inputs and outputs have already been shuffled and augmented.
		"""

		num_examples = inputs.shape[self._batch_axis]
		if num_examples <= 0:
			raise ValueError('Invalid number of examples')

		num_steps = ((num_examples - 1) // self._batch_size + 1) if num_examples > 0 else 0
		if num_steps <= 0:
			raise ValueError('Invalid number of steps')

		batch_inputs_dict, batch_outputs_dict = dict(), dict()
		num_saved_examples = 0
		for step in range(num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			# FIXME [fix] >> Does not work correctly in time-major data.
			batch_inputs, batch_outputs = inputs[start:end], outputs[start:end]
			if batch_inputs.size > 0 and batch_outputs.size > 0 and len(batch_inputs) == len(batch_outputs):  # If batch_inputs and batch_outputs are non-empty.
				#if self._augmenter is not None:
				#	batch_inputs, batch_outputs = self._augmenter(batch_inputs, batch_outputs, self._is_output_augmented)
				batch_name = 'batch_{}'.format(step)
				batch_inputs_dict[batch_name], batch_outputs_dict[batch_name] = batch_inputs, batch_outputs
				num_saved_examples += len(batch_inputs)

		return batch_inputs_dict, batch_outputs_dict, num_saved_examples

	@staticmethod
	def _loadDataFromNpyFiles(input_filepaths, output_filepaths, batch_axis):
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
		inputs, outputs = None, None
		for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
			inp, outp = np.load(input_filepath), np.load(output_filepath)
			if inp.shape[batch_axis] != outp.shape[batch_axis]:
				raise ValueError('Unmatched shapes of {} and {}'.format(input_filepath, output_filepath))
			inputs = inp if inputs is None else np.concatenate((inputs, inp), axis=0)
			outputs = outp if outputs is None else np.concatenate((outputs, outp), axis=0)

		return inputs, outputs

	@staticmethod
	def _loadDataFromImageFiles(input_filepaths, batch_axis):
		inputs = list()
		for input_filepath in input_filepaths:
			inp = cv.imread(input_filepath)
			inputs.append(inp)

		return np.array(inputs)

#--------------------------------------------------------------------

class NpzFileBatchGeneratorFromNpyFiles(NpzFileBatchGeneratorFromFiles):
	"""Loads data from npy files, generates their batches and saves them to npz files.
	"""

	def __init__(self, input_filepaths, output_filepaths, num_loaded_files, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None):
		"""
		In this constructor, no data will be loaded from input and output npy files.

		Inputs:
			input_filepaths (a list of strings): A list of input npy files.
			output_filepaths (a list of strings): A list of output npy files.
			num_loaded_files (int): The number of files that can be loaded at a time.
			batch_size (int): Batch size.
			shuffle (bool): Specifies whether data is shuffled or not.
			is_time_major (bool): Specifies whether inputs is time-major or batch-major.
			augmenter (object): Data augmenter. It can be None.
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
			is_output_augmented (bool): Specifies whether outputs is augmented or not.
			batch_input_filename_format (string): A file name format to save batch inputs. batch_input_filename_format = 'batch_input_{}.npz' when it is None.
			batch_output_filename_format (string): A file name format to save batch outputs. batch_output_filename_format = 'batch_output_{}.npz' when it is None.
			batch_info_csv_filename (string): A CSV file name to save info about batch inputs and outputs. batch_info_csv_filename = 'batch_info.csv' when it is None.
		"""

		super().__init__(num_loaded_files, batch_size, shuffle, is_time_major, augmenter, is_output_augmented, batch_input_filename_format, batch_output_filename_format, batch_info_csv_filename)

		if input_filepaths is None or output_filepaths is None:
			raise ValueError('input_filepaths or output_filepaths should not be None')
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')

		"""
		# TODO [enhance] >> When there are many files, this part is too slow.
		for input_filepath, label_filepath in zip(input_filepaths, output_filepaths):
			inp, outp = np.load(output_filepath), np.load(output_filepath)
			if inp.shape[self._batch_axis] != outp.shape[self._batch_axis]:
				raise ValueError('Unmatched shapes of {} and {}'.format(input_filepath, output_filepath))
		"""

		self._num_files = len(input_filepaths)
		self._num_file_groups = ((self._num_files - 1) // self._num_loaded_files + 1) if self._num_files > 0 else 0
		if self._num_file_groups <= 0:
			raise ValueError('Invalid number of file groups')

		self._input_filepaths, self._output_filepaths = np.array(input_filepaths), np.array(output_filepaths)

	def saveBatches(self, dir_path, *args, **kwargs):
		"""Saves batches.

		Inputs:
			dir_path (string): A directory path to save batches.
		Outputs:
			The number of saved examples.
		"""

		file_indices = np.arange(self._num_files)
		if self._shuffle:
			np.random.shuffle(file_indices)

		total_saved_example_count, start_batch_file_idx = 0, 0
		saved_file_info = list()
		for gid in range(self._num_file_groups):
			start = gid * self._num_loaded_files
			end = start + self._num_loaded_files
			sub_file_indices = file_indices[start:end]
			if sub_file_indices.size > 0:  # If sub_file_indices is non-empty.
				sub_input_filepaths, sub_output_filepaths = self._input_filepaths[sub_file_indices], self._output_filepaths[sub_file_indices]
				if sub_input_filepaths.size > 0 and sub_output_filepaths.size > 0:  # If sub_input_filepaths and sub_output_filepaths are non-empty.
					inputs, outputs = NpzFileBatchGeneratorFromNpyFiles._loadDataFromNpyFiles(sub_input_filepaths, sub_output_filepaths, self._batch_axis)

					sub_saved_file_info, num_saved_examples = self._saveBatchesToNpzFiles(inputs, outputs, dir_path, sub_file_indices.size, start_batch_file_idx)
					saved_file_info.extend(sub_saved_file_info)
					total_saved_example_count += num_saved_examples
					start_batch_file_idx += sub_file_indices.size

		with open(os.path.join(dir_path, self._batch_info_csv_filename), mode='w', encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for info in saved_file_info:
				writer.writerow(info)

		return total_saved_example_count

#--------------------------------------------------------------------

class NpzFileBatchGeneratorFromImageFiles(NpzFileBatchGeneratorFromFiles):
	"""Loads data from images files, generates their batches and saves them to npz files.
	"""

	def __init__(self, input_filepaths, output_seqs, num_loaded_files, batch_size, shuffle=True, is_time_major=False, augmenter=None, is_output_augmented=False, batch_input_filename_format=None, batch_output_filename_format=None, batch_info_csv_filename=None):
		"""
		In this constructor, no data will be loaded from input image files.

		Inputs:
			input_filepaths (a list of strings): A list of input image files.
			output_seqs (a list of strings or integers): A list of output sequences (e.g. labels).
			num_loaded_files (int): The number of files that can be loaded at a time.
			batch_size (int): Batch size.
			shuffle (bool): Specifies whether data is shuffled or not.
			is_time_major (bool): Specifies whether inputs is time-major or batch-major.
			augmenter (object): Data augmenter. It can be None.
				inputs, outputs = augmenter(inputs, outputs, is_output_augmented).
			is_output_augmented (bool): Specifies whether outputs is augmented or not.
			batch_input_filename_format (string): A file name format to save batch inputs. batch_input_filename_format = 'batch_input_{}.npz' when it is None.
			batch_output_filename_format (string): A file name format to save batch outputs. batch_output_filename_format = 'batch_output_{}.npz' when it is None.
			batch_info_csv_filename (string): A CSV file name to save info about batch inputs and outputs. batch_info_csv_filename = 'batch_info.csv' when it is None.
		"""

		super().__init__(num_loaded_files, batch_size, shuffle, is_time_major, augmenter, is_output_augmented, batch_input_filename_format, batch_output_filename_format, batch_info_csv_filename)

		if input_filepaths is None or output_seqs is None:
			raise ValueError('input_filepaths or output_seqs should not be None')
		if len(input_filepaths) != len(output_seqs):
			raise ValueError('Unmatched lengths of input_filepaths and output_seqs')

		"""
		# TODO [enhance] >> When there are many files, this part is too slow.
		for input_filepath, output in zip(input_filepaths, output_seqs):
			inp, outp = np.load(input_filepath), np.load(output)
			if inp.shape[self._batch_axis] != outp.shape[self._batch_axis]:
				raise ValueError('Unmatched shapes of {} and {}'.format(input_filepath, output))
		"""

		self._num_files = len(input_filepaths)
		self._num_file_groups = ((self._num_files - 1) // self._num_loaded_files + 1) if self._num_files > 0 else 0
		if self._num_file_groups <= 0:
			raise ValueError('Invalid number of file groups')

		self._input_filepaths, self._output_seqs = np.array(input_filepaths), np.array(output_seqs)

	def saveBatches(self, dir_path, *args, **kwargs):
		"""Saves batches.

		Inputs:
			dir_path (string): A directory path to save batches.
		Outputs:
			The number of saved examples.
		"""

		file_indices = np.arange(self._num_files)
		if self._shuffle:
			np.random.shuffle(file_indices)

		total_saved_example_count, start_batch_file_idx = 0, 0
		saved_file_info = list()
		for gid in range(self._num_file_groups):
			start = gid * self._num_loaded_files
			end = start + self._num_loaded_files
			sub_file_indices = file_indices[start:end]
			if sub_file_indices.size > 0:  # If sub_file_indices is non-empty.
				sub_input_filepaths, outputs = self._input_filepaths[sub_file_indices], self._output_seqs[sub_file_indices]
				if sub_input_filepaths.size > 0 and outputs.size > 0:  # If sub_input_filepaths and outputs are non-empty.
					inputs = NpzFileBatchGeneratorFromImageFiles._loadDataFromImageFiles(sub_input_filepaths, self._batch_axis)

					#num_files = sub_file_indices.size
					num_files = 1
					sub_saved_file_info, num_saved_examples = self._saveBatchesToNpzFiles(inputs, outputs, dir_path, num_files, start_batch_file_idx)
					saved_file_info.extend(sub_saved_file_info)
					total_saved_example_count += num_saved_examples
					start_batch_file_idx += num_files

		with open(os.path.join(dir_path, self._batch_info_csv_filename), mode='w', encoding='UTF8', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for info in saved_file_info:
				writer.writerow(info)

		return total_saved_example_count
