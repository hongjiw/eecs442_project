import os
def rc_init(bucket_size, forecast_depth, mode):
	rc_path = {}
	params = {}
	
	params['bucket_size'] = bucket_size;
	params['forecast_depth'] = forecast_depth;
	params['mode'] = mode
	params['save'] = True
	params['retrain'] = True
	params['visual'] = False
	params['gpu'] = False


	rc_path['save_pred'] = '/home/hongjiw/research/data/RC/clips/pred_loc.txt'
	
	rc_path['solver'] = './rc_solver_' + mode + '.prototxt'
	rc_path['network'] = './rc_test_' + mode + '.prototxt'

	#used for generating network files
	rc_path['solver_template'] = './rc_solver_' + mode + '_template.prototxt'
	rc_path['network_template'] = './rc_test_' + mode + '_template.prototxt'

	rc_path['train_list_template'] = './train_list.txt'
	rc_path['test_list_template'] = './test_list.txt'

	#setup the test data file
	rc_path['test_file'] = '/home/hongjiw/research/data/RC/clips/test_b_' + \
						str(params['bucket_size']) + '_f_' + str(params['forecast_depth']) + \
						'_' + params['mode'] + '.h5'
	
	if not os.path.isfile(rc_path['test_file']):
		print("Missing the input file")

	return rc_path, params