# REF [site] >> https://github.com/fchollet/keras/issues/3338

import numpy as np

def center_crop(x, center_crop_size, **kwargs):
	centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
	halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
	return x[:, centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
	np.random.seed(sync_seed)
	w, h = x.shape[1], x.shape[2]
	rangew = (w - random_crop_size[0]) // 2
	rangeh = (h - random_crop_size[1]) // 2
	offsetw = 0 if rangew == 0 else np.random.randint(rangew)
	offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
	return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]
