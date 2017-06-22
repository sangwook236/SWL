import os
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
os.chdir(swl_python_home_dir_path + '/test/machine_learning')

import sys
sys.path.append('../../src')

#%%------------------------------------------------------------------

import numpy as np
import keras
import swl

#%%------------------------------------------------------------------
# to_one_hot_encoding().

y = np.array([ 1, 3, 5, 2, 4, 0 ])

np.set_printoptions(threshold=np.nan)

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y), sep='\n')
print('keras.utils.to_categorical ->', keras.utils.to_categorical(y), sep='\n')

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y, 10), sep='\n')
print('keras.utils.to_categorical ->', keras.utils.to_categorical(y, 10), sep='\n')

Y = np.reshape(y, [2, 3, 1])

print('Y ->', Y.shape)

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(Y).shape)
print(swl.machine_learning.util.to_one_hot_encoding(Y))
print('keras.utils.to_categorical ->', keras.utils.to_categorical(Y).shape)
print(keras.utils.to_categorical(Y).reshape(Y.shape[:-1] + (-1,)))

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(Y, 10).shape)
print(swl.machine_learning.util.to_one_hot_encoding(Y, 10))
print('keras.utils.to_categorical ->', keras.utils.to_categorical(Y, 10).shape)
print(keras.utils.to_categorical(Y, 10).reshape(Y.shape[:-1] + (-1,)))
