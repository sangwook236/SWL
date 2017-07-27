class Dataset(object):
	def __init__(self, data, labels):
		assert data.shape[0] == labels.shape[0], 'Invalid sizes of data and labels'
		self._data = data
		self._labels = labels
		self._num_examples = data.shape[0]

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self._data = data

	@property
	def labels(self):
		return self._labels

	@labels.setter
	def labels(self, labels):
		self._labels = labels

	@property
	def num_examples(self):
		return self._num_examples
