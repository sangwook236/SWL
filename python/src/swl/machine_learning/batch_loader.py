import os, abc, csv
import numpy as np

#%%------------------------------------------------------------------
# FileBatchLoader.
#	Loads batches from files.
class FileBatchLoader(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a generator.
	@abc.abstractmethod
	def loadBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

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
				num_examples = int(row[2])
				yield batch_inputs, batch_outputs, num_examples
