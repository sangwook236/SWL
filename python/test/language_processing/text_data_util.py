import os
import cv2
import PIL.Image

# REF [function] >> FileBasedTextDatasetBase._load_data_from_image_label_info() in text_data.py
def load_data_from_image_label_info(label_converter, image_label_info_filepath, image_channel, max_label_len, image_label_separator=' ', is_pil=False):
	# In a image-label info file:
	#	Each line consists of 'image-filepath + image-label-separator + label'.

	try:
		with open(image_label_info_filepath, 'r', encoding='UTF8') as fd:
			#lines = fd.readlines()  # A list of strings.
			lines = fd.read().splitlines()  # A list of strings.
	except FileNotFoundError as ex:
		print('[SWL] Error: File not found: {}.'.format(image_label_info_filepath))
		raise
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error: {}.'.format(image_label_info_filepath))
		raise

	if 1 == image_channel:
		flag = cv2.IMREAD_GRAYSCALE
	elif 3 == image_channel:
		flag = cv2.IMREAD_COLOR
	elif 4 == image_channel:
		flag = cv2.IMREAD_ANYCOLOR  # ?
	else:
		flag = cv2.IMREAD_UNCHANGED

	dir_path = os.path.dirname(image_label_info_filepath)
	images, labels_str, labels_int = list(), list(), list()
	for line in lines:
		img_fpath, label_str = line.split(image_label_separator, 1)

		if len(label_str) > max_label_len:
			print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
			continue
		fpath = os.path.join(dir_path, img_fpath)
		img = cv2.imread(fpath, flag)
		if img is None:
			print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
			continue

		#if resize_functor:
		#	img = resize_functor(img, image_height, image_width)
		try:
			label_int = label_converter.encode(label_str)  # Decorated/undecorated label ID.
		except Exception:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			continue
		if label_str != label_converter.decode(label_int):
			print('[SWL] Warning: Mismatched original and decoded labels: {} != {}.'.format(label_str, label_converter.decode(label_int)))
			# TODO [check] >> I think such data should be used to deal with unknown characters (as negative data) in real data.
			#continue

		images.append(PIL.Image.fromarray(img) if is_pil else img)
		labels_str.append(label_str)
		labels_int.append(label_int)

	#if preprocess_functor:
	#	images, labels_int = preprocess_functor(images, labels_int)

	return images, labels_str, labels_int

# REF [function] >> FileBasedTextDatasetBase._load_data_from_image_and_label_files() in text_data.py
def load_data_from_image_and_label_files(label_converter, image_filepaths, label_filepaths, image_channel, max_label_len, is_pil=False):
	if len(image_filepaths) != len(label_filepaths):
		print('[SWL] Error: Different lengths of image and label files, {} != {}.'.format(len(image_filepaths), len(label_filepaths)))
		return
	for img_fpath, lbl_fpath in zip(image_filepaths, label_filepaths):
		img_fname, lbl_fname = os.path.splitext(os.path.basename(img_fpath))[0], os.path.splitext(os.path.basename(lbl_fpath))[0]
		if img_fname != lbl_fname:
			print('[SWL] Warning: Different file names of image and label pair, {} != {}.'.format(img_fname, lbl_fname))
			continue

	if 1 == image_channel:
		flag = cv2.IMREAD_GRAYSCALE
	elif 3 == image_channel:
		flag = cv2.IMREAD_COLOR
	elif 4 == image_channel:
		flag = cv2.IMREAD_ANYCOLOR  # ?
	else:
		flag = cv2.IMREAD_UNCHANGED

	images, labels_str, labels_int = list(), list(), list()
	for img_fpath, lbl_fpath in zip(image_filepaths, label_filepaths):
		try:
			with open(lbl_fpath, 'r', encoding='UTF8') as fd:
				#label_str = fd.read()
				#label_str = fd.read().rstrip()
				label_str = fd.read().rstrip('\n')
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found: {}.'.format(lbl_fpath))
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error: {}.'.format(lbl_fpath))
			continue
		if len(label_str) > max_label_len:
			print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
			continue
		img = cv2.imread(img_fpath, flag)
		if img is None:
			print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
			continue

		#if resize_functor:
		#	img = resize_functor(img, image_height, image_width)
		try:
			label_int = label_converter.encode(label_str)  # Decorated/undecorated label ID.
		except Exception:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			continue
		if label_str != label_converter.decode(label_int):
			print('[SWL] Warning: Mismatched original and decoded labels: {} != {}.'.format(label_str, label_converter.decode(label_int)))
			# TODO [check] >> I think such data should be used to deal with unknown characters (as negative data) in real data.
			#continue

		images.append(PIL.Image.fromarray(img) if is_pil else img)
		labels_str.append(label_str)
		labels_int.append(label_int)

	#if preprocess_functor:
	#	images, labels_int = preprocess_functor(images, labels_int)

	return images, labels_str, labels_int
