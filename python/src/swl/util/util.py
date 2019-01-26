import os, re, csv
import numpy as np

def find_most_frequent_value(arr):
	"""
	Args:
	    arr: 1-D array.
	"""
	counts = np.bincount(arr)
	return np.argmax(counts)

def top_k_values(iterable, k):
	# In ascending order.
	#return sorted(iterable)[-k:]  # Top-k values.
	# In descending order.
	return sorted(iterable, reverse=True)[:k]  # Top-k values.

def bottom_k_values(iterable, k):
	# In ascending order.
	return sorted(iterable)[:k]  # Bottom-k values.
	# In descending order.
	#return sorted(iterable, reverse=True)[-k:]  # Bottom-k values.

def top_k_indices(iterable, k):
	# In ascending order.
	#return sorted(range(len(iterable)), key=lambda i: iterable[i])[-k:]  # Top-k indices.
	# In descending order.
	return sorted(range(len(iterable)), key=lambda i: iterable[i], reverse=True)[:k]  # Top-k indices.

def bottom_k_indices(iterable, k):
	# In ascending order.
	return sorted(range(len(iterable)), key=lambda i: iterable[i])[:k]  # Bottom-k indices.
	# In descending order.
	#return sorted(range(len(iterable)), key=lambda i: iterable[i], reverse=True)[-k:]  # Bottom-k indices.

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def load_npy_files_in_directory(dir_path, file_prefix, file_suffix):
	file_extension = 'npy'
	arr_list = list()
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search('^' + file_prefix, filename) and re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					arr = np.load(filepath)
					arr_list.append(arr)
			break  # Do not include subdirectories.
	return arr_list

def load_data_from_npy_files(input_filepaths, output_filepaths, batch_axis=0):
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

def load_filepaths_from_npy_file_info(npy_file_csv_filepath):
	input_filepaths, output_filepaths, example_counts = list(), list(), list()
	with open(npy_file_csv_filepath, 'r', encoding='UTF8') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if not row:
				continue
			input_filepaths.append(row[0])
			output_filepaths.append(row[1])
			example_counts.append(int(row[2]))
	return input_filepaths, output_filepaths, example_counts

def save_data_to_npy_files(inputs, outputs, num_files, dir_path, input_filename_format, output_filename_format, npy_file_csv_filename, batch_axis=0, start_file_index=0, mode='w'):
	num_examples = inputs.shape[batch_axis]
	if num_examples <= 0:
		raise ValueError('Invalid number of examples')
	num_examples_in_a_file = ((num_examples - 1) // num_files + 1) if num_examples > 0 else 0
	if num_examples_in_a_file <= 0:
		raise ValueError('Invalid number of examples in a file')

	make_dir(dir_path)

	indices = np.arange(num_examples)
	np.random.shuffle(indices)

	with open(os.path.join(dir_path, npy_file_csv_filename), mode=mode, encoding='UTF8', newline='') as csvfile:
		writer = csv.writer(csvfile)

		for file_step in range(num_files):
			start = file_step * num_examples_in_a_file
			end = start + num_examples_in_a_file
			data_indices = indices[start:end]
			if data_indices.size > 0:  # If data_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				sub_inputs, sub_outputs = inputs[data_indices], outputs[data_indices]
				if sub_inputs.size > 0 and sub_outputs.size > 0:  # If sub_inputs and sub_outputs are non-empty.
					input_filepath, output_filepath = os.path.join(dir_path, input_filename_format.format(start_file_index + file_step)), os.path.join(dir_path, output_filename_format.format(start_file_index + file_step))
					np.save(input_filepath, sub_inputs)
					np.save(output_filepath, sub_outputs)
					writer.writerow((input_filepath, output_filepath, len(data_indices)))

def shuffle_data_in_npy_files(input_filepaths, output_filepaths, num_loaded_files, num_shuffles, tmp_dir_path_prefix=None, shuffle_input_filename_format=None, shuffle_output_filename_format=None, shuffle_info_csv_filename=None, is_time_major=False):
	"""
	input_filepaths: A list of input npy files.
	output_filepaths: A list of output npy files.
	num_shuffles: The number of shuffles.
	num_loaded_files: The number of files that can be loaded at one time.
	tmp_dir_path_prefix: A prefix for temporary directoy paths where shuffled data are saved.
	is_time_major: Data is time-major or batch-major.
	"""
	
	batch_axis = 1 if is_time_major else 0
	if tmp_dir_path_prefix is None:
		tmp_dir_path_prefix = 'shuffle_tmp_dir_{}'
	if shuffle_input_filename_format is None:
		shuffle_input_filename_format = 'shuffle_input_{}.npy'
	if shuffle_output_filename_format is None:
		shuffle_output_filename_format = 'shuffle_output_{}.npy'
	if shuffle_info_csv_filename is None:
		shuffle_info_csv_filename = 'shuffle_info.csv'

	for shuffle_step in range(num_shuffles):
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')

		num_files = len(input_filepaths)
		num_file_groups = ((num_files - 1) // num_loaded_files + 1) if num_files > 0 else 0
		if num_file_groups <= 0:
			raise ValueError('Invalid number of file groups')

		file_indices = np.arange(num_files)
		np.random.shuffle(file_indices)

		dir_path = tmp_dir_path_prefix.format(shuffle_step)
		start_file_index = 0
		for gid in range(num_file_groups):
			start = gid * num_loaded_files
			end = start + num_loaded_files
			sub_file_indices = file_indices[start:end]
			if sub_file_indices.size > 0:  # If sub_file_indices is non-empty.
				sub_input_filepaths, sub_output_filepaths = input_filepaths[sub_file_indices], output_filepaths[sub_file_indices]
				if sub_input_filepaths.size > 0 and sub_output_filepaths.size > 0:  # If sub_input_filepaths and sub_output_filepaths are non-empty.
					inputs, outputs = load_data_from_npy_files(sub_input_filepaths, sub_output_filepaths, batch_axis)
					save_data_to_npy_files(inputs, outputs, num_loaded_files, dir_path, shuffle_input_filename_format, shuffle_output_filename_format, shuffle_info_csv_filename, batch_axis, start_file_index, 'w' if 0 == gid else 'a')
					start_file_index += num_loaded_files
		input_filepaths, output_filepaths, _ = load_filepaths_from_npy_file_info(os.path.join(dir_path, shuffle_info_csv_filename))
		input_filepaths, output_filepaths = np.array(input_filepaths), np.array(output_filepaths)
