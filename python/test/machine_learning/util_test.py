import os
os.chdir('D:/work/swl_github/python/test/image_processing')

import sys
sys.path.append('../../src')

#%%------------------------------------------------------------------

import numpy as np
import keras
import swl

#%%------------------------------------------------------------------

y = [ 1, 3, 5, 2, 4, 0 ]

np.set_printoptions(threshold=np.nan)

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y), sep='\n')
print('keras.utils.to_categorical ->', keras.utils.to_categorical(y), sep='\n')

print('swl.machine_learning.util.to_one_hot_encoding ->', swl.machine_learning.util.to_one_hot_encoding(y, 10), sep='\n')
print('keras.utils.to_categorical ->', keras.utils.to_categorical(y, 10), sep='\n')
