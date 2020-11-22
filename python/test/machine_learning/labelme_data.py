import os, math, itertools, functools, json, glob
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image
import cv2

#--------------------------------------------------------------------

# REF [site] >> https://github.com/wkentaro/labelme
class LabelMeDataset(torch.utils.data.Dataset):
	def __init__(self, json_filepaths, image_channel):
		super().__init__()

		"""
		version
		flags
		shapes *
			label
			line_color
			fill_color
			points
			group_id
			shape_type
		lineColor
		fillColor
		imagePath
		imageData
		imageWidth
		imageHeight
		"""

		#self.data_dicts = self._load_data_from_json(json_filepaths, image_channel)
		self.data_dicts = self._load_data_from_json_async(json_filepaths, image_channel)

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		return self.data_dicts[idx]

	def _load_data_from_json(self, json_filepaths, image_channel):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		data_dicts = list()
		for json_filepath in json_filepaths:
			try:
				with open(json_filepath, 'r') as fd:
					json_data = json.load(fd)
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
				continue

			try:
				version = json_data['version']
				flags = json_data['flags']
				line_color, fill_color = json_data['lineColor'], json_data['fillColor']

				dir_path = os.path.dirname(json_filepath)
				image_filepath = os.path.join(dir_path, json_data['imagePath'])
				image_data = json_data['imageData']
				image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

				img = cv2.imread(image_filepath, flag)
				if img is None:
					print('[SWL] Error: Failed to load an image, {}.'.format(image_filepath))
					continue

				shapes = list()
				for shape in json_data['shapes']:
					label, points, group_id, shape_type = shape['label'], shape['points'], shape['group_id'], shape['shape_type']
					#shape_line_color, shape_fill_color = shape['line_color'], shape['fill_color']
					try:
						shape_line_color = shape['line_color']
					except KeyError as ex:
						#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
						shape_line_color = None
					try:
						shape_fill_color = shape['fill_color']
					except KeyError as ex:
						#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
						shape_fill_color = None
					shape_dict = {
						'label': label,
						'line_color': shape_line_color,
						'fill_color': shape_fill_color,
						'points': points,
						'group_id': group_id,
						'shape_type': shape_type,
					}
					shapes.append(shape_dict)

				data_dict = {
					'version': version,
					'flags': flags,
					'shapes': shapes,
					'lineColor': line_color,
					'fillColor': fill_color,
					'imagePath': image_filepath,
					'imageData': image_data,
					'imageWidth': image_width,
					'imageHeight': image_height,
				}
				data_dicts.append(data_dict)
			except KeyError as ex:
				print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
				data_dicts.append(None)

		return data_dicts

	def _load_data_from_json_async(self, json_filepaths, image_channel):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		#--------------------
		async_results = list()
		def async_callback(result):
			# This is called whenever sqr_with_sleep(i) returns a result.
			# async_results is modified only by the main process, not the pool workers.
			async_results.append(result)

		num_processes = 8
		#timeout = 10
		timeout = None
		#with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		with mp.Pool(processes=num_processes) as pool:
			#results = pool.map_async(functools.partial(self._worker_proc, flag=flag), json_filepaths)
			results = pool.map_async(functools.partial(self._worker_proc, flag=flag), json_filepaths, callback=async_callback)

			results.get(timeout)

		data_dicts = list(res for res in async_results[0] if res is not None)
		return data_dicts

	def _worker_proc(self, json_filepath, flag):
		try:
			with open(json_filepath, 'r') as fd:
				json_data = json.load(fd)
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
			return None
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
			return None

		try:
			version = json_data['version']
			flags = json_data['flags']
			line_color, fill_color = json_data['lineColor'], json_data['fillColor']

			dir_path = os.path.dirname(json_filepath)
			image_filepath = os.path.join(dir_path, json_data['imagePath'])
			image_data = json_data['imageData']
			image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

			if True:
				img = cv2.imread(image_filepath, flag)
				if img is None:
					print('[SWL] Error: Failed to load an image, {}.'.format(image_filepath))
					return None

			shapes = list()
			for shape in json_data['shapes']:
				label, points, group_id, shape_type = shape['label'], shape['points'], shape['group_id'], shape['shape_type']
				#shape_line_color, shape_fill_color = shape['line_color'], shape['fill_color']
				try:
					shape_line_color = shape['line_color']
				except KeyError as ex:
					#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
					shape_line_color = None
				try:
					shape_fill_color = shape['fill_color']
				except KeyError as ex:
					#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
					shape_fill_color = None
				shape_dict = {
					'label': label,
					'line_color': shape_line_color,
					'fill_color': shape_fill_color,
					'points': points,
					'group_id': group_id,
					'shape_type': shape_type,
				}
				shapes.append(shape_dict)

			return {
				'version': version,
				'flags': flags,
				'shapes': shapes,
				'lineColor': line_color,
				'fillColor': fill_color,
				'imagePath': image_filepath,
				'imageData': image_data,
				'imageWidth': image_width,
				'imageHeight': image_height,
			}
		except KeyError as ex:
			print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
			return None

#--------------------------------------------------------------------

class FigureLabelMeDataset(torch.utils.data.Dataset):
	def __init__(self, pkl_filepath, image_channel, classes, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__()

		self.classes = classes
		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform
		self.data_dir_path = os.path.dirname(pkl_filepath)

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		# Load figure infos from a pickle file.
		import pickle
		try:
			with open(pkl_filepath, 'rb') as fd:
				self.figures, self.labels = pickle.load(fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pkl_filepath, ex))
			self.figures, self.labels = None, None
		assert (self.figures is None and self.labels is None) or len(self.figures) == len(self.labels)

	def __len__(self):
		return len(self.figures)

	def __getitem__(self, idx):
		if self.is_preloaded_image_used:
			image = Image.fromarray(self.figures[idx])
		else:
			img_fpath, rct = self.figures[idx]  # (left, top, right, bottom).
			img_fpath = os.path.join(self.data_dir_path, img_fpath)
			try:
				image = Image.open(img_fpath).crop(rct)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(img_fpath, ex))
				image = None

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label

	def get_class(self, label_id):
		return self.classes[label_id]

#--------------------------------------------------------------------

# REF [class] >> PennFudanDataset in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_object_detection.py.
class FigureDetectionLabelMeDataset(torch.utils.data.Dataset):
	def __init__(self, pkl_filepath, image_channel, classes, is_preloaded_image_used=True, transform=None):
		super().__init__()

		self.classes = classes
		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.data_dir_path = os.path.dirname(pkl_filepath)

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		# Load figure infos from a pickle file.
		import pickle
		try:
			with open(pkl_filepath, 'rb') as fd:
				self.figures = pickle.load(fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pkl_filepath, ex))
			self.figures = None

	def __len__(self):
		return len(self.figures)

	def __getitem__(self, idx):
		image, boxes, labels = self.figures[idx]
		assert len(boxes) == len(labels)

		if self.is_preloaded_image_used:
			image = Image.fromarray(image)
		else:
			img_fpath = os.path.join(self.data_dir_path, image)
			try:
				image = Image.open(img_fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(img_fpath, ex))
				image = None

		num_objs = len(boxes)
		# Keypoint visibility:
		#	If visibility == 0, a keypoint is not in the image.
		#	If visibility == 1, a keypoint is in the image but not visible, namely maybe behind of an object.
		#	if visibility == 2, a keypoint looks clearly, not hidden.
		keypoints = np.array([[[box[0], box[1], 2], [box[0], box[3], 2], [box[2], box[3], 2], [box[2], box[1], 2]] for box in boxes])

		"""
		boxes = torch.tensor(boxes, dtype=torch.float32)
		keypoints = torch.tensor(keypoints, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.int64)

		image_id = torch.tensor([idx])
		area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
		# Suppose all instances are not crowd.
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
		"""
		image_id = np.array([idx])
		area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
		# Suppose all instances are not crowd.
		iscrowd = np.zeros((num_objs,), dtype=np.int64)

		target = {
			'boxes': boxes,
			#'masks': masks,
			'keypoints': keypoints,
			'labels': labels,
			'image_id': image_id,
			'area': area,
			'iscrowd': iscrowd,
		}

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image, target = self.transform(image, target)

		return image, target

	def get_class(self, label_id):
		return self.classes[label_id]
