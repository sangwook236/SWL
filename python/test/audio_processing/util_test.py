import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#--------------------
import swl

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

wav_dir_path = dataset_home_dir_path + '/failure_analysis/defect/knock_sound/Wave_75sample'

#%%------------------------------------------------------------------

wav_suffix = ''

wavs, freqs = swl.audio_processing.util.load_wave_files(wav_dir_path, wav_suffix)
