#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time
import cv2
from swl.language_processing import synthtext_dataset

# REF [file] >> ${ssd_detectors_HOME}/data_synthtext.py
def load_synthtext_data_info_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	#synthtext_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/synth_text_vgg/SynthText'
	synthtext_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/SynthText/SynthText'
	gt_filepath = gt_path = os.path.join(synthtext_dir_path, 'gt.mat')

	print('Start loading SynthText dataset...')
	start_time = time.time()
	image_names, word_bboxes, char_bboxes, gt_texts = synthtext_dataset.load_synthtext_data_info(synthtext_dir_path, gt_filepath)
	print('End loading SynthText dataset: {} secs.'.format(time.time() - start_time))

	#--------------------
	for image_name, word_boxes, char_boxes, texts in zip(image_names, word_bboxes, char_bboxes, gt_texts):
		img = cv2.imread(os.path.join(synthtext_dir_path, image_name))
		img_height, img_width = img.shape[:2]

		word_boxes = word_boxes.reshape(word_boxes.shape[0], -1, 2)
		char_boxes = char_boxes.reshape(char_boxes.shape[0], -1, 2)
		word_boxes[:,:,0] *= img_width
		word_boxes[:,:,1] *= img_height
		char_boxes[:,:,0] *= img_width
		char_boxes[:,:,1] *= img_height
		word_boxes = np.round(word_boxes).astype(np.int)
		char_boxes = np.round(char_boxes).astype(np.int)

		print('GT texts =', texts)
		rgb = img.copy()
		for box in word_boxes:
			if False:
				cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
			else:
				cv2.line(rgb, tuple(box[0,:]), tuple(box[1,:]), (0, 0, 255), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[1,:]), tuple(box[2,:]), (0, 255, 0), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[2,:]), tuple(box[3,:]), (255, 0, 0), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[3,:]), tuple(box[0,:]), (255, 0, 255), 2, cv2.LINE_8)

		cv2.imshow('SynthText', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	load_synthtext_data_info_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
