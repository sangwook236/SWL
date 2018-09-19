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
