import numpy as np

def standardize_samplewise(data):
	for idx in range(data.shape[0]):
		for ch in range(data.shape[3]):
			mean = np.mean(data[idx,:,:,ch])
			sd = np.std(data[idx,:,:,ch])
			if np.isclose(sd, 0.0) == True or np.isnan(sd) == True:
				#print('[Warning] sd = 0')
				data[idx,:,:,ch] -= mean
				#data[idx,:,:,ch] = 0
			else:
				data[idx,:,:,ch] = (data[idx,:,:,ch] - mean) / sd
	return data

def standardize_featurewise(data):
	for r in range(data.shape[1]):
		for c in range(data.shape[2]):
			mean = np.mean(data[:,r,c,:], axis=0)
			sd = np.std(data[:,r,c,:], axis=0)
			if np.any(np.isclose(sd, np.zeros(sd.size))) == True or np.any(np.isnan(sd)) == True:
				#print('[Warning] sd = 0')
				for ch in range(data.shape[3]):
					if np.isclose(sd[ch], 0.0) == True or np.isnan(sd[ch]) == True:
						data[:,r,c,ch] -= mean[ch]
						#data[:,r,c,ch] = 0
					else:
						data[:,r,c,ch] = (data[:,r,c,ch] - mean[ch]) / sd[ch]
			else:
				data[:,r,c,:] = (data[:,r,c,:] - mean) / sd
	return data

def normalize_samplewise_by_min_max(data):
	for idx in range(data.shape[0]):
		for ch in range(data.shape[3]):
			dmin = np.amin(data[idx,:,:,ch])
			dmax = np.amax(data[idx,:,:,ch])
			if np.isclose(dmin, dmax) == True:
				#print('[Warning] max - min = 0')
				data[idx,:,:,ch] -= dmin
				#data[idx,:,:,ch] = 0
			else:
				data[idx,:,:,ch] = (data[idx,:,:,ch] - dmin) / (dmax - dmin)
	return data

def normalize_featurewise_by_min_max(data):
	for r in range(data.shape[1]):
		for c in range(data.shape[2]):
			dmin = np.amin(data[:,r,c,:], axis=0)
			dmax = np.amax(data[:,r,c,:], axis=0)
			if np.any(np.isclose(dmin, dmax)) == True:
				#print('[Warning] max - min = 0')
				for ch in range(data.shape[3]):
					if np.isclose(dmin[ch], dmax[ch]) == True:
						data[:,r,c,ch] -= dmin[ch]
						#data[:,r,c,ch] = 0
					else:
						data[:,r,c,ch] = (data[:,r,c,ch] - dmin[ch]) / (dmax[ch] - dmin[ch])
			else:
				data[:,r,c,:] = (data[:,r,c,:] - dmin) / (dmax - dmin)
	return data
