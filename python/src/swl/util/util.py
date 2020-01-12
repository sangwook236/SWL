import os, math, re, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

#--------------------------------------------------------------------

def add_mask(mask, sub_mask, bbox=None):
	if bbox is None:
		bbox = (0, 0, sub_mask.shape[1], sub_mask.shape[0])

	mask_pil = Image.fromarray(mask)
	text_mask_pil = Image.fromarray(sub_mask)
	mask_pil.paste(text_mask_pil, bbox, text_mask_pil)
	mask = np.asarray(mask_pil, dtype=mask.dtype)
	"""
	tmp_mask_pil = Image.fromarray(np.zeros_like(mask))
	tmp_mask_pil.paste(Image.fromarray(sub_mask), bbox)
	tmp_mask = np.asarray(tmp_mask_pil, dtype=mask.dtype)

	mask = np.where(mask >= tmp_mask, mask, tmp_mask)
	"""

	return mask

def add_pdf(pdf, sub_pdf, bbox=None):
	if bbox is None:
		bbox = (0, 0, sub_pdf.shape[1], sub_pdf.shape[0])

	tmp_pdf_pil = Image.fromarray(np.zeros_like(pdf))
	tmp_pdf_pil.paste(Image.fromarray(sub_pdf), bbox)
	tmp_pdf = np.asarray(tmp_pdf_pil, dtype=pdf.dtype)

	pdf = np.where(pdf >= tmp_pdf, pdf, tmp_pdf)
	#pdf += tmp_pdf
	#pdf /= np.sum(pdf)  # sum(pdf) = 1.

	return pdf

def draw_using_mask(img, mask, color, bbox=None):
	if 2 == img.ndim:
		text_colored = np.round(mask * (color / 255))
	elif 3 == img.ndim:
		text_colored = np.zeros(mask.shape + (len(color),), dtype=img.dtype)
		for ch in range(img.shape[-1]):
			text_colored[...,ch] = np.round(mask * (color[ch] / 255))
	else:
		raise ValueError('Invalid image dimension: {}'.format(img.ndim))

	if bbox is None:
		bbox = (0, 0, mask.shape[1], mask.shape[0])

	img_pil = Image.fromarray(img)
	img_pil.paste(Image.fromarray(text_colored), bbox, Image.fromarray(mask))
	img = np.asarray(img_pil, dtype=img.dtype)

	return img

#--------------------------------------------------------------------

def download(url, output_dir_path):
	import urllib.request
	response = urllib.request.urlopen(url)

	file_names = url.split('/')[-1].split('.')
	filename, fileext = file_names[0], file_names[-1]
	#if 'zip' == fileext or fileext.find('zip') != -1:
	if 'zip' == fileext:
		import zipfile
		response = zipfile.ZipFile(response)
		content = zipfile.ZipFile.open(response).read()

		import os
		output_dir_path = os.path.join(output_dir_path, filename)
		with open(output_dir_path, 'w') as fd:
			fd.write(content.read())
	#elif 'gz' == fileext or fileext.find('gz') != -1:
	elif 'gz' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:gz', fileobj=response)
		tar.extractall(output_dir_path)
	#elif 'bz2' == fileext or fileext.find('bz2') != -1 or 'bzip2' == fileext or fileext.find('bzip2') != -1:
	elif 'bz2' == fileext or 'bzip2' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:bz2', fileobj=response)
		tar.extractall(output_dir_path)
	#elif 'xz' == fileext or fileext.find('xz') != -1:
	elif 'xz' == fileext:
		import tarfile
		tar = tarfile.open(mode='r:xz', fileobj=response)
		tar.extractall(output_dir_path)
	else:
		raise ValueError('Unexpected file extention, {}'.format(fileext))
		return False

	return True

#--------------------------------------------------------------------

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def list_files_in_directory(dir_path, file_prefix, file_suffix, file_extension, is_recursive=False):
	filepaths = list()
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search('^' + file_prefix, filename) and re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepaths.append(os.path.join(root, filename))
			if not is_recursive:
				break  # Do not include subdirectories.
	return filepaths

def extract_subset_of_data(data, subset_ratio):
	"""
	Inputs:
		data (numpy.array): Data.
		subset_ratio (float): The ratio of subset of data. 0.0 < subset_ratio <= 1.0.
	"""

	num_data = len(data)
	num_sub_data = math.ceil(num_data * subset_ratio)
	indices = np.arange(num_data)
	np.random.shuffle(indices)
	return data[indices[:num_sub_data]]

#--------------------------------------------------------------------

def load_filepaths_from_npy_file_info(npy_file_csv_filepath):
	input_filepaths, output_filepaths, example_counts = list(), list(), list()
	# Input-filepath,output-filepath,example-count.
	with open(npy_file_csv_filepath, 'r', encoding='UTF8') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if not row:
				continue
			input_filepaths.append(row[0])
			output_filepaths.append(row[1])
			example_counts.append(int(row[2]))
	return input_filepaths, output_filepaths, example_counts

def load_filepaths_from_npz_file_info(npy_file_csv_filepath):
	return load_filepaths_from_npy_file_info(npy_file_csv_filepath)

def load_data_from_npy_files(input_filepaths, output_filepaths, batch_axis=0):
	if len(input_filepaths) != len(output_filepaths):
		raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')
	inputs, outputs = None, None
	for image_filepath, label_filepath in zip(input_filepaths, output_filepaths):
		inp, outp = np.load(image_filepath), np.load(label_filepath)
		if inp.shape[batch_axis] != outp.shape[batch_axis]:
			raise ValueError('Unmatched shapes of {} and {}'.format(image_filepath, label_filepath))
		inputs = inp if inputs is None else np.concatenate((inputs, inp), axis=0)
		outputs = outp if outputs is None else np.concatenate((outputs, outp), axis=0)
	return inputs, outputs

def load_data_from_npz_file(npz_filepath, batch_axis=0, input_name_format='input_{}', output_name_format='output_{}'):
	npzfile = np.load(npz_filepath)

	input_keys = [key for key in npzfile.keys() if input_name_format.format('') in key]
	output_keys = [key for key in npzfile.keys() if output_name_format.format('') in key]
	#input_keys = sorted([key for key in npzfile.keys() if input_name_format.format('') in key])
	#output_keys = sorted([key for key in npzfile.keys() if output_name_format.format('') in key])
	if len(input_keys) != len(output_keys):
		raise ValueError('The numbers of inputs and outputs are different: {} != {}.'.format(len(input_keys), len(output_keys)))

	inputs, outputs = None, None
	for ink, outk in zip(input_keys, output_keys):
		inp, outp = npzfile[ink], npzfile[outk]
		if inp.shape[batch_axis] != outp.shape[batch_axis]:
			raise ValueError("Unmatched shapes of input '{}' and output '{}'".format(ink, outk))
		inputs = inp if inputs is None else np.concatenate((inputs, inp), axis=0)
		outputs = outp if outputs is None else np.concatenate((outputs, outp), axis=0)
	return inputs, outputs

def save_data_to_npy_files(inputs, outputs, save_dir_path, num_files, input_filename_format, output_filename_format, npy_file_csv_filename, batch_axis=0, start_file_index=0, mode='w', shuffle=True):
	num_examples = inputs.shape[batch_axis]
	if outputs.shape[batch_axis] != num_examples:
		raise ValueError('The number of inputs is not equal to the number of outputs')
	if num_examples <= 0:
		raise ValueError('Invalid number of examples')
	num_examples_in_a_file = ((num_examples - 1) // num_files + 1) if num_examples > 0 else 0
	if num_examples_in_a_file <= 0:
		raise ValueError('Invalid number of examples in a file')

	make_dir(save_dir_path)

	indices = np.arange(num_examples)
	if shuffle:
		np.random.shuffle(indices)

	input_filepaths, output_filepaths, data_lens = list(), list(), list()
	for file_step in range(num_files):
		start = file_step * num_examples_in_a_file
		end = start + num_examples_in_a_file
		data_indices = indices[start:end]
		if data_indices.size > 0:  # If data_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			sub_inputs, sub_outputs = inputs[data_indices], outputs[data_indices]
			if sub_inputs.size > 0 and sub_outputs.size > 0:  # If sub_inputs and sub_outputs are non-empty.
				input_filepath, output_filepath = os.path.join(save_dir_path, input_filename_format.format(start_file_index + file_step)), os.path.join(save_dir_path, output_filename_format.format(start_file_index + file_step))
				np.save(input_filepath, sub_inputs)
				np.save(output_filepath, sub_outputs)
				input_filepaths.append(input_filepath)
				output_filepaths.append(output_filepath)
				data_lens.append(len(data_indices))

	with open(os.path.join(save_dir_path, npy_file_csv_filename), mode=mode, encoding='UTF8', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for input_filepath, output_filepath, data_len in zip(input_filepaths, output_filepaths, data_lens):
			writer.writerow((input_filepath, output_filepath, data_len))

	return input_filepaths, output_filepaths, data_lens

def save_data_to_npz_file(inputs, outputs, npz_filepath, num_examples_in_a_file, shuffle=True, batch_axis=0, input_name_format='input_{}', output_name_format='output_{}'):
	if num_examples_in_a_file <= 0:
		raise ValueError('Invalid number of examples in a file')
	num_examples = inputs.shape[batch_axis]
	if outputs.shape[batch_axis] != num_examples:
		raise ValueError('The number of inputs is not equal to the number of outputs')
	if num_examples <= 0:
		raise ValueError('Invalid number of examples')

	indices = np.arange(num_examples)
	if shuffle:
		np.random.shuffle(indices)

	dataset = dict()
	sub_idx, start_idx = 0, 0
	while True:
		end_idx = start_idx + num_examples_in_a_file
		data_indices = indices[start_idx:end_idx]
		if data_indices.size > 0:  # If data_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			sub_inputs, sub_outputs = inputs[data_indices], outputs[data_indices]
			if sub_inputs.size > 0 and sub_outputs.size > 0:  # If sub_inputs and sub_outputs are non-empty.
				dataset[input_name_format.format(sub_idx)], dataset[output_name_format.format(sub_idx)] = sub_inputs, sub_outputs
				sub_idx += 1

		if end_idx >= num_examples:
			break;
		start_idx = end_idx
			
	np.savez(npz_filepath, **dataset)

def shuffle_data_in_npy_files(input_filepaths, output_filepaths, num_files_loaded_at_a_time, num_shuffles, tmp_dir_path_prefix=None, shuffle_input_filename_format=None, shuffle_output_filename_format=None, shuffle_info_csv_filename=None, is_time_major=False):
	"""
	Inputs:
		input_filepaths (a list of strings): A list of input npy files.
		output_filepaths (a list of strings): A list of output npy files.
		num_shuffles (int): The number of shuffles to run.
		num_files_loaded_at_a_time (int): The number of files that can be loaded at a time.
		tmp_dir_path_prefix (string): A prefix for temporary directoy paths where shuffled data are saved.
		is_time_major (bool): Data is time-major or batch-major.
	"""
	
	batch_axis = 1 if is_time_major else 0
	if tmp_dir_path_prefix is None:
		tmp_dir_path_prefix = 'shuffle_tmp_dir_{}'
	if shuffle_input_filename_format is None:
		shuffle_input_filename_format = 'shuffle_input_{}.npy'
	if shuffle_output_filename_format is None:
		shuffle_output_filename_format = 'shuffle_output_{}.npy'
	if shuffle_info_csv_filename is None:
		shuffle_info_csv_filename = 'shuffle_file_info.csv'

	for shuffle_step in range(num_shuffles):
		if len(input_filepaths) != len(output_filepaths):
			raise ValueError('Unmatched lengths of input_filepaths and output_filepaths')

		num_files = len(input_filepaths)
		num_file_groups = ((num_files - 1) // num_files_loaded_at_a_time + 1) if num_files > 0 else 0
		if num_file_groups <= 0:
			raise ValueError('Invalid number of file groups')

		file_indices = np.arange(num_files)
		np.random.shuffle(file_indices)

		save_dir_path = tmp_dir_path_prefix.format(shuffle_step)
		start_file_index = 0
		for gid in range(num_file_groups):
			start = gid * num_files_loaded_at_a_time
			end = start + num_files_loaded_at_a_time
			sub_file_indices = file_indices[start:end]
			if sub_file_indices.size > 0:  # If sub_file_indices is non-empty.
				sub_input_filepaths, sub_output_filepaths = input_filepaths[sub_file_indices], output_filepaths[sub_file_indices]
				if sub_input_filepaths.size > 0 and sub_output_filepaths.size > 0:  # If sub_input_filepaths and sub_output_filepaths are non-empty.
					inputs, outputs = load_data_from_npy_files(sub_input_filepaths, sub_output_filepaths, batch_axis)
					save_data_to_npy_files(inputs, outputs, num_files_loaded_at_a_time, save_dir_path, shuffle_input_filename_format, shuffle_output_filename_format, shuffle_info_csv_filename, batch_axis, start_file_index, 'w' if 0 == gid else 'a')
					start_file_index += num_files_loaded_at_a_time
		input_filepaths, output_filepaths, _ = load_filepaths_from_npy_file_info(os.path.join(save_dir_path, shuffle_info_csv_filename))
		input_filepaths, output_filepaths = np.array(input_filepaths), np.array(output_filepaths)
