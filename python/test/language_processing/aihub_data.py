import os, json, time
import numpy as np
import torch
import cv2

#--------------------------------------------------------------------

# REF [site] >> http://www.aihub.or.kr/ai_data
#	#syllables = 558,600, #words = 277,150, #sentences = 42,350.
class AiHubPrintedTextDataset(torch.utils.data.Dataset):
	def __init__(self, label_converter, json_filepath, data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__()

		self._label_converter = label_converter
		self.data_dir_path = data_dir_path
		self.image_height, self.image_width, self.image_channel = image_height, image_width, image_channel
		self.max_label_len = max_label_len
		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform

		self.mode = 'RGB'

		self.images, self.labels_str, self.labels_int = self._load_data_from_json(json_filepath, data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		from PIL import Image

		if self.is_preloaded_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
				image = None
		target = [self.label_converter.pad_id] * (self.max_label_len + self.label_converter.num_affixes)
		label_ext_int = self.labels_int[idx]  # Decorated/undecorated integer label.
		target_len = len(label_ext_int)
		target[:target_len] = label_ext_int

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

	@property
	def label_converter(self):
		return self._label_converter

	@property
	def shape(self):
		return self.image_height, self.image_width, self.image_channel

	# REF [function] >> FileBasedTextLineDatasetBase._load_data_from_image_and_label_files() in text_line_data.py.
	def _load_data_from_json(self, json_filepath, data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used=True):
		try:
			with open(json_filepath, encoding='UTF8') as fd:
				json_data = json.load(fd)
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(json_filepath))
			return None, None, None
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(json_filepath))
			return None, None, None

		image_types = {'글자(음절)': 'syllable', '단어(어절)': 'word', '문장': 'sentence'}
		additional_data_dir_path = '01_printed_{}_images'

		print('[SWL] Info: Start loading AI Hub dataset...')
		start_time = time.time()
		image_infos = dict()
		for info in json_data['images']:
			image_infos[int(info['id'])] = {'width': info['width'], 'height': info['height'], 'file_name': info['file_name']}

		image_dict = dict()
		for info in json_data['annotations']:
			img_type = image_types[info['attributes']['type']]
			if img_type not in image_types_to_load: continue

			img_id = int(info['image_id'])
			image_dict[img_id] = {**image_infos[img_id], 'label': info['text']}
			image_dict[img_id]['file_name'] = os.path.join(additional_data_dir_path.format(img_type), image_infos[img_id]['file_name'])
		print('[SWL] Info: End loading AI Hub dataset: {} secs.'.format(time.time() - start_time))
		del image_infos

		print('[SWL] Info: Start generating a dataset...')
		start_time = time.time()
		images, labels_str, labels_int = list(), list(), list()
		for info in image_dict.values():
			img_fname, label_str = info['file_name'], info['label']
			img_height, img_width = info['height'], info['width']

			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			if self.is_preloaded_image_used:
				fpath = os.path.join(data_dir_path, img_fname)
				img = cv2.imread(fpath)
				if img is None:
					print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
					continue

				#assert img.shape[0] == img_height and img.shape[1] == img_width
			else: img = None

			#img = self.resize(img, None, image_height, image_width)
			try:
				label_int = self.label_converter.encode(label_str)  # Decorated/undecorated integer label.
			except Exception:
				print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.label_converter.decode(label_int):
				print('[SWL] Error: Mismatched original and decoded labels: {} != {}.'.format(label_str, self.label_converter.decode(label_int)))
				# TODO [check] >> I think such data should be used to deal with unknown characters (as negative data) in real data.
				#continue

			images.append(img if is_preloaded_image_used else img_fname)
			labels_str.append(label_str)
			labels_int.append(label_int)
		print('[SWL] Info: End generating a dataset: {} secs.'.format(time.time() - start_time))

		##images = list(map(lambda image: self.resize(image), images))
		#images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
		#images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int
