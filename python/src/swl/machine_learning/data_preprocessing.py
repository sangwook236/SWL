import numpy as np

def featurewise_std_normalization(data):
	for r in range(data.shape[1]):
		for c in range(data.shape[2]):
			mean = np.mean(data[:,r,c,:], axis=0)
			sd = np.std(data[:,r,c,:], axis=0)
			if np.any(np.isclose(sd, np.zeros(sd.size))) == True or np.any(np.isnan(sd)) == True:
				#print('[Warning] sd = 0')
				for ch in range(data.shape[3]):
					if np.isclose(sd[ch], 0.0) == True or np.isnan(sd[ch]) == True:
						data[:,r,c,ch] -= mean[ch]
					else:
						data[:,r,c,ch] = (data[:,r,c,ch] - mean[ch]) / sd[ch]
			else:
				data[:,r,c,:] = (data[:,r,c,:] - mean) / sd
	return data

def samplewise_std_normalization(data):
	for idx in range(data.shape[0]):
		for ch in range(data.shape[3]):
			mean = np.mean(data[idx,:,:,ch])
			sd = np.std(data[idx,:,:,ch])
			if np.isclose(sd, 0.0) == True or np.isnan(sd) == True:
				#print('[Warning] sd = 0')
				data[idx,:,:,ch] -= mean
			else:
				data[idx,:,:,ch] = (data[idx,:,:,ch] - mean) / sd
	return data
