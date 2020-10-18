#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch

def save_model(model_filepath, model, logger=None):
	#torch.save(model.state_dict(), model_filepath)
	torch.save({'state_dict': model.state_dict()}, model_filepath)
	if logger: logger.info('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model, device='cpu'):
	loaded_data = torch.load(model_filepath, map_location=device)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	print('Loaded a model from {}.'.format(model_filepath))
	return model

def build_model():
	# TODO [implement] >>
	raise NotImplementedError

# Model averaging:
#	The paper averages the last k checkpoints to create an ensembling effect.
def average_models(model, models):
	for ps in zip(*[mdl.params() for mdl in [model] + models]):
		ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

def simple_model_averaging_example():
	gpu = -1
	device = torch.device(('cuda:{}'.format(gpu) if gpu >= 0 else 'cuda') if torch.cuda.is_available() else 'cpu')

	model_filepaths = [
		'./model_01.pth',
		'./model_02.pth'
	]
	averaged_model_filepath = './averaged_model.pth'

	models = list()
	for mdl_fpath in model_filepaths:
		mdl = build_model()
		models.append(load_model(mdl_fpath, mdl, device))

	averaged_model = build_model()  # Averaged model.
	average_models(averaged_model, models)

	save_model(averaged_model_filepath, averaged_model)

	inputs = None
	predictions = averaged_model(inputs)

def average_predictions(models, inputs):
	predictions = list(mdl(inputs) for mdl in models)
	predictions = np.array(predictions)
	return np.average(predictions, axis=0)
	#return np.sum(predictions, axis=0)

def simple_prediction_averaging_example():
	gpu = -1
	device = torch.device(('cuda:{}'.format(gpu) if gpu >= 0 else 'cuda') if torch.cuda.is_available() else 'cpu')

	model_filepaths = [
		'./model_01.pth',
		'./model_02.pth'
	]

	models = list()
	for mdl_fpath in model_filepaths:
		mdl = build_model()
		models.append(load_model(mdl_fpath, mdl, device))

	inputs = None
	predictions = average_predictions(models, inputs)

def main():
	simple_model_averaging_example()
	simple_prediction_averaging_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
