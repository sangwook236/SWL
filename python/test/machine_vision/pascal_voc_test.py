#!/usr/bin/env python
# coding: UTF-8

import os, glob
import pascal_voc_tools
import cv2

def load_pascal_voc_format_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/Reciept_1'
	save_dir_path = data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line'

	xml_filepaths = sorted(glob.glob(os.path.join(data_dir_path, '*.xml')))
	if not xml_filepaths:
		print('Error: No xml files.')
		return

	parser = pascal_voc_tools.XmlParser()

	os.makedirs(save_dir_path, exist_ok=False)
	save_file_id = 0
	for xml_filepath in xml_filepaths:
		elements = parser.load(xml_filepath)
		if not elements:
			print('Error: Failed to load a XML file, {}.'.format(xml_filepath))
			continue

		#folder = elements['folder']
		image_filepath = elements['filename']
		image_shape = elements['size']['height'], elements['size']['width'], elements['size']['depth']
		objects = elements['object']

		image_filepath = os.path.join(data_dir_path, image_filepath)
		img = cv2.imread(image_filepath)
		if img is None:
			print('Error: Failed to load an image file, {}.'.format(image_filepath))
			continue

		for obj in objects:
			name = obj['name']
			bbox = obj['bndbox']

			if name:
				x1, y1, x2, y2 = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
				patch = img[y1:y2+1,x1:x2+1]

				patch_fpath = os.path.join(save_dir_path, 'file_{:06}.png'.format(save_file_id))
				cv2.imwrite(patch_fpath, patch)
				label_fpath = os.path.join(save_dir_path, 'file_{:06}.txt'.format(save_file_id))
				try:
					with open(label_fpath, 'w', encoding='UTF8') as fd:
						fd.write(name)
				except FileNotFoundError as ex:
					print('File not found: {}.'.format(label_fpath))
				except UnicodeDecodeError as ex:
					print('Unicode decode error: {}.'.format(label_fpath))

				save_file_id += 1
			else:
				print('Warning: Null name in bbox ({}, {}, {}, {}).'.format(int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])))

def main():
	load_pascal_voc_format_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
