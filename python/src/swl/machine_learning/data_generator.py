import numpy as np
import threading

def create_dataset_generator_from_array(X, Y, batch_size, shuffle=False):
	num_steps = np.ceil(len(X) / batch_size).astype(np.int)
	if shuffle is True:
		indexes = np.arange(len(X))
		np.random.shuffle(indexes)
		for idx in range(num_steps):
			batch_x = X[indexes[idx*batch_size:(idx+1)*batch_size]]
			batch_y = Y[indexes[idx*batch_size:(idx+1)*batch_size]]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)
	else:
		for idx in range(num_steps):
			batch_x = X[idx*batch_size:(idx+1)*batch_size]
			batch_y = Y[idx*batch_size:(idx+1)*batch_size]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)

# NOTICE [info] >> This is not thread-safe. To make it thread-safe, use ThreadSafeGenerator.
def create_dataset_generator_using_imgaug(seq, X, Y, batch_size, shuffle=True, dataset_preprocessing_function=None, num_classes=None):
	while True:
		seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		X_aug = seq_det.augment_images(X)
		Y_aug = seq_det.augment_images(Y)

		# Preprocessing (normalization, standardization, etc).
		if dataset_preprocessing_function is not None:
			X_aug, Y_aug = dataset_preprocessing_function(X_aug, Y_aug, num_classes)

		num_steps = np.ceil(len(X_aug) / batch_size).astype(np.int)
		#num_steps = len(X_aug) // batch_size + (0 if len(X_aug) % batch_size == 0 else 1)
		if shuffle is True:
			indexes = np.arange(len(X_aug))
			np.random.shuffle(indexes)
			for idx in range(num_steps):
				batch_x = X_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				batch_y = Y_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				#yield {'input': batch_x}, {'output': batch_y}
				yield batch_x, batch_y
		else:
			for idx in range(num_steps):
				batch_x = X_aug[idx*batch_size:(idx+1)*batch_size]
				batch_y = Y_aug[idx*batch_size:(idx+1)*batch_size]
				#yield {'input': batch_x}, {'output': batch_y}
				yield batch_x, batch_y

class DatasetGeneratorUsingImgaug:
	def __init__(self, seq, X, Y, num_classes, batch_size, shuffle=True, dataset_preprocessing_function=None):
		self.seq = seq
		self.X = X
		self.Y = Y
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dataset_preprocessing_function = dataset_preprocessing_function

		self.num_steps = np.ceil(len(self.X) / self.batch_size).astype(np.int)
		#self.num_steps = len(self.X) // self.batch_size + (0 if len(self.X) % self.batch_size == 0 else 1)
		self.idx = 0
		self.X_aug = None
		self.Y_aug = None

		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			if 0 == self.idx:
				seq_det = self.seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
				self.X_aug = seq_det.augment_images(self.X)
				self.Y_aug = seq_det.augment_images(self.Y)

				# Preprocessing (normalization, standardization, etc).
				if self.dataset_preprocessing_function is not None:
					self.X_aug, self.Y_aug = self.dataset_preprocessing_function(self.X_aug, self.Y_aug, self.num_classes)

				indexes = np.arange(len(self.X_aug))
				if self.shuffle is True:
					np.random.shuffle(indexes)

			if self.X_aug is None or self.Y_aug is None:
				assert False, 'Both X_aug and Y_aug are not None.'

			if self.shuffle is True:
				batch_x = self.X_aug[indexes[self.idx*self.batch_size:(self.idx+1)*self.batch_size]]
				batch_y = self.Y_aug[indexes[self.idx*self.batch_size:(self.idx+1)*self.batch_size]]
			else:
				batch_x = self.X_aug[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
				batch_y = self.Y_aug[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
			self.idx = (self.idx + 1) % self.num_steps
			return batch_x, batch_y
