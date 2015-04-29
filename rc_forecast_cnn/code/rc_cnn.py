#set up caffe module
import sys
import caffe
caffe_root = '/home/hongjiw/research/library/caffe'
sys.path.insert(0, caffe_root + 'python')


def update_solver():
	print "none"

def update_train_test():
	print "none"

def update_network(rc_path, params):
	template = open(rc_path['network_template'], "rt").read()

	with open(rc_path['network'], "wt") as output:
		output.write (template % params)

def cnn_train(solver_file_path):
	solver = caffe.get_solver(solver_file_path)
	solver.solve()
	solver.test_nets[0].forward()
	loss = solver.test_nets[0].blobs['loss'].data
	print "Test loss is: %f (Modify the prototxt to enable batch test)" %(loss)

def vis_square(data, title, padsize=1, padval=0):
	data -= data.min()
	data /= data.max()
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	
	#show the image
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(title, 400, 400)
	cv2.imshow(title, data)
	cv2.waitKey(0)

def cnn_netinfo(network_file_path, pretrained_model_path):
	#check if the file exists
	if not os.path.isfile(pretrained_model_path and network_file_path):
		print("Missing the input file")

	#initialize the caffe model
	net = caffe.Net(network_file_path, pretrained_model_path, caffe.TEST)
	
	#layer features and there shapes
	print [(k, v.data.shape) for k, v in net.blobs.items()]
	print [(k, v[0].data.shape) for k, v in net.params.items()]

	#the parameters are a list of [weights, biases]
	#display learned weights
	for k, v in net.params.items():
		print k, v
		feat = net.params[k][0].data
		if len(feat.shape) == 2:
			fig = plt.figure()
			fig.suptitle(k, fontsize=14, fontweight='bold')
			plt.subplot(2, 1, 1)
			plt.plot(feat.flat)
			plt.subplot(2, 1, 2)
			_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
			plt.show(k)
		elif len(feat.shape) == 4:
			vis_square(feat.transpose(0, 2, 3, 1), k)

def rc_cnn_predict(net, data, rn, outlayer):
	#make sure the batch size and the saved hdf5 size are equal 
	#print data.shape
	#print net.blobs['data'].data.shape
	assert(net.blobs['data'].data.shape == data.shape)

	#pre-assign the input blob
	net.blobs['data'].data[...] = data	
	input_blob_buffer = net.blobs['data'].data[...]
	
	#recurrent forecast
	pred = np.array([])
	for rn_ind in range(0,rn):
		#forward pass the network
		net.forward()

		#collect the prediction
		loc = net.blobs[outlayer].data
		if not pred.any():
			pred = np.copy(loc)
		else:
			pred = np.concatenate((pred, loc), axis=1)
		#update the input blob for recurrent prediction
		if rn > 1:
			input_blob_prev = np.split(input_blob_buffer, [1], axis=3)[1]
			input_blob_new = np.split(input_blob_buffer, [1], axis=3)[0]
			input_blob_new = np.reshape(loc, input_blob_new.shape)
			input_blob = np.concatenate((input_blob_prev, input_blob_new), axis=3)
			net.blobs['data'].data[...] = np.copy(input_blob)
			input_blob_buffer = np.copy(input_blob)

	#return the forecast loss
	return pred

def cnn_main(params, rc_path):
	#use CPU modeNone
	if params['gpu']:
		caffe.set_mode_gpu()

	#train the caffe
	if params['retrain']:
		cnn_train(rc_path['solver'])

	#show the network infomation
	if params['visual']:
		cnn_netinfo(rc_path['network'], rc_path['model'])

	#initialize the caffe model
	net = caffe.Net(rc_path['network'], rc_path['model'], caffe.TEST)
	
	#read in data
	if not os.path.isfile(rc_path['test_file']):
		print("Missing the input file")
	data, label = hdf5_read(rc_path['test_file'])

	#rc prediction
	pred = rc_cnn_predict(net, data, params['fc_depth'], 'fc1')

	#save the predicted locations
	if params['save']:
		np.savetxt(rc_path['save_pred'], pred, fmt='%f', delimiter=',')
		print "Prediction result saved to: %s" %(rc_path['save_pred'])

	#calculate loss
	loss = rc_loss(pred, label)
	print "Forecast loss with %d forecast depth is: %f" %(params['fc_depth'], loss)

	#analysis
	loss_stack = np.array([])
	
	for rn in range(1, params['fc_depth']+1):
		pred = rc_cnn_predict(net, data, rn, 'fc1')
		loss = rc_loss(pred, label)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, params['fc_depth']+1):
		print "Loss with %d forecast depth(cnn rec1): %f" %(rn, loss_stack[rn-1])	
	return loss_stack