import numpy as np

def find_most_frequent_value(arr):
	"""
	Args:
	    arr: 1-D array.
	"""
	counts = np.bincount(arr)
	return np.argmax(counts)
