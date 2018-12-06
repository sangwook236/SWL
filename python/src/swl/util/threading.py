import threading

# REF [site] >> http://anandology.com/blog/using-iterators-and-generators/
class ThreadSafeIterator:
	"""
	Takes an iterator/generator and makes it thread-safe by serializing call to the 'next' method of given iterator/generator.
	"""
	def __init__(self, it):
		self._it = it
		self._lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self._lock:
			return self._it.next()

class ThreadSafeGenerator:
	"""
	Takes a generator and makes it thread-safe by serializing call to the 'next' method of given iterator/generator.
	"""
	def __init__(self, gen):
		self._gen = gen
		self._lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self._lock:
			return next(self._gen)
