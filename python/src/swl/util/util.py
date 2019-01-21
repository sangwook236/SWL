import os, re
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
