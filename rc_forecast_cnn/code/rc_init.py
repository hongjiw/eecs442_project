import os
from rc_cnn import *
from rc_bvlc_extract import *

def rc_init(dev_path, data_path, bucket_size, forecast_depth, mode):
	rc_path = {}
	params = {}
	
	params['bucket_size'] = bucket_size;
	params['forecast_depth'] = forecast_depth;
	params['mode'] = mode
	params['save'] = True
	params['retrain'] = False
	params['visual'] = False
	params['gpu'] = True
	params['radius'] = 1.5
	params['resize'] = 227
	params['output_layer'] = 'fc2'

	rc_path['solver'] = dev_path + '/rc_solver_' + mode + '.prototxt'
	rc_path['train_list'] = dev_path + '/train_list.txt'
	rc_path['test_list'] = dev_path + '/test_list.txt'
	rc_path['data_train_list'] = data_path + '/train_list.txt'
	rc_path['data_trainval_list'] = data_path + '/trainval_list.txt'
	rc_path['data_test_list'] = data_path + '/test_list.txt'
	rc_path['train_prototxt'] = dev_path + '/rc_train_' + mode + '.prototxt'
	rc_path['test_prototxt'] = dev_path + '/rc_test_' + mode + '.prototxt'

	#used for generating network files
	rc_path['solver_template'] = dev_path + '/rc_solver_' + mode + '_template.prototxt'
	rc_path['train_list_template'] = dev_path + '/train_list_template.txt'
	rc_path['test_list_template'] = dev_path + '/test_list_template.txt'
	rc_path['train_prototxt_template'] = dev_path + '/rc_train_' + mode + '_template.prototxt'
	rc_path['test_prototxt_template'] = dev_path +  '/rc_test_' + mode + '_template.prototxt'

	#setup the test data file
	rc_path['test_file'] = data_path + '/test_b_' + \
						str(params['bucket_size']) + '_f_' + str(params['forecast_depth']) + \
						'_' + mode + '.h5'

	rc_path['train_file'] = data_path + '/train_b_' + \
						str(params['bucket_size']) + '_f_' + str(params['forecast_depth']) + \
						'_' + mode + '.h5'

	rc_path['trainval_file'] = data_path + '/trainval_b_' + \
						str(params['bucket_size']) + '_f_' + str(params['forecast_depth']) + \
						'_' + mode + '.h5'
							
	rc_path['model'] = dev_path + '/../model/b_' + str(bucket_size) + '_f_' \
					+str(forecast_depth) + '_' + mode + '_iter_5000.caffemodel'

	rc_path['dev_path'] = dev_path
	rc_path['data_path'] = data_path
	rc_path['save_pred'] = data_path + '/pred_loc_b_' + str(bucket_size) \
						+ '_f_' + str(forecast_depth) + '_' + mode + '.txt'
	
	rc_path['bvlc_model'] = '/home/hongjiw/research/library/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

	if not os.path.isfile(rc_path['test_file']) and mode == 'ALEX':
		#make sure to set the path correctly
		rc_path['alex_feature_filename'] = 'alex_feature.mat'
		rc_path['bvlc_test'] = '/home/hongjiw/research/library/caffe/models/bvlc_alexnet/test.prototxt'
		rc_path['bvlc_test_template'] = '/home/hongjiw/research/library/caffe/models/bvlc_alexnet/test_template.prototxt'
		rc_bvlc_extract(rc_path, params)

	return rc_path, params