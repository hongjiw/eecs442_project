import numpy as np
import scipy as sp
import shutil
import tempfile
import matplotlib.pyplot as plt
import io
import cv2

import os
import sys
	
from rc_baseline import *
from rc_init import *
from rc_cnn import *

if __name__ == '__main__':
	
	#setup basic parameters
	bucket_size = 25;
	forecast_depth = 50;
	mode = 'MO'

	rc_path, params = rc_init(bucket_size, forecast_depth, mode)

	#read in data
	data, label = hdf5_read(rc_path['test_file'])

	#baselines
	baseline_main(data, label, params['forecast_depth'])

	#cnn 
	update_network(rc_path, {"input_dim": data.shape[0]})
	update_train_test()

	
	#rc_path['model'] = '../model/rec_1_cfl_iter_10000.caffemodel'
	
	#cnn_loss_stack = cnn_main(params, rc_path)

