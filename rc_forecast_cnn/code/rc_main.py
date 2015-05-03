import numpy as np
import matplotlib.pyplot as plt
import io

import os
import sys
	
from rc_baseline import *
from rc_init import *
from rc_cnn import *
from misan import *

if __name__ == '__main__':
	
	#path
	dev_path = '/home/hongjiw/research/eecs442_project/rc_forecast_cnn/code'
	data_path = '/home/hongjiw/research/data/RC/clips'

	#record all the prediction results
	logfile_path = dev_path + '/log.txt'
	if os.path.isfile(logfile_path):
		os.remove(logfile_path)

	#caffe fearure
	mode = 'ALEX'

	#motion feature
	#mode = 'MO'

	for bucket_size in range(25, 76, 25):
		for forecast_depth in range(25, 76, 25):
			with open(logfile_path, "a") as logfile:
				logfile.write('\n')
				logfile.write("bucket_size: " + str(bucket_size) + '\n')
				logfile.write("forecast_depth " + str(forecast_depth) + '\n')
				logfile.write("mode: " + str(mode) + '\n')
			
			#initialize parameters(path) for CNN
			rc_path, params = rc_init(dev_path, data_path, bucket_size, forecast_depth, mode)

			#read in data
			data, label = hdf5_read(rc_path['test_file'])

			#baselines
			baseline_main(data, label, params['forecast_depth'], logfile_path)

			#cnn
			for wd in [0.0005]:#frange(0.0005, 0.005, 0.001):
				for lr in [0.0001]:#frange(0.00001, 0.0001, 0.00001): 0.00039

					#log the learning rate and weight decay
					with open(logfile_path, "a") as logfile:
						logfile.write("lr: " + str(lr) + '\n')
						logfile.write("wd: " + str(wd) + '\n')

					#update the cnn network files
					template_params = {"dev_path": dev_path, "data_path": data_path, 
						"b": bucket_size, "f": forecast_depth, "mode": mode,
						"input_num": label.shape[0], "output_dim": label.shape[1],
						"lr": lr, "wd": wd
					}
					
					update_template(rc_path['solver_template'], rc_path['solver'], template_params)
					update_template(rc_path['test_prototxt_template'], rc_path['test_prototxt'], 
							template_params)
					update_template(rc_path['train_prototxt_template'], 
							rc_path['train_prototxt'], template_params)
					update_template(rc_path['train_list_template'], 
							rc_path['train_list'], template_params)
					update_template(rc_path['test_list_template'], 
							rc_path['test_list'], template_params)


					#cnn pipeline
					cnn_loss = cnn_main(data, label, params, rc_path, logfile_path)