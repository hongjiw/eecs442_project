#set up caffe module
import sys
import numpy as np
import caffe
from rc_metric import *

caffe_root = '/home/hongjiw/research/library/caffe'
sys.path.insert(0, caffe_root + 'python')

def cnn_train(solver_file_path, logfile_path):

	solver = caffe.get_solver(solver_file_path)
	solver.solve()
	solver.test_nets[0].forward()
	loss = solver.test_nets[0].blobs['loss'].data
	print "Test loss is: %f (Modify the prototxt to enable batch test)" %(loss)
	#log this
	with open(logfile_path, "a") as logfile:
		logfile.write('cnn trainval loss: ' + str(loss) + '\n')

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
	
	#recurrent forecast (but we are only using single forecast for now)
	net.forward()

	#collect the prediction
	loc = net.blobs[outlayer].data
	pred = np.copy(loc)
	#return the forecast loss
	return pred

def cnn_main(data, label, params, rc_path, logfile_path):
	
	#use CPU modeNone
	if params['gpu']:
		caffe.set_mode_gpu()

	#train the caffe
	if params['retrain']:
		cnn_train(rc_path['solver'], logfile_path)

	#show the network infomation
	if params['visual']:
		cnn_netinfo(rc_path['test_prototxt'], rc_path['model'])

	#initialize the caffe model
	net = caffe.Net(rc_path['test_prototxt'], rc_path['model'], caffe.TEST)
	
	#rc prediction
	pred = rc_cnn_predict(net, data, 1, params['output_layer'])
	
	#save the predicted locations
	if params['save']:
		np.savetxt(rc_path['save_pred'], pred, fmt='%f', delimiter=',')
		print "Prediction result saved to: %s" %(rc_path['save_pred'])

	#calculate loss
	loss = rc_loss(pred, label)

	#report the loss
	print 'cnn test loss: ' + str(loss);

	#log this
	with open(logfile_path, "a") as logfile:
		logfile.write('cnn test loss: ' + str(loss) + '\n')

	return loss