import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import shutil
import tempfile
import matplotlib.pyplot as plt
import h5py
import io
import cv2

#set up the root path for caffe
caffe_root = '/home/hongjiw/research/library/caffe'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#check if the model file exists
import os

def rc_loss(pred, label):
	#pred is a N x (2*rn) numpy matrix
	#label is a N x 2 numpy matrix

	#make sure input is valid	
	assert(not pred.shape[1] % 2)
	assert(pred.shape[0] == label.shape[0])

	#rn: recurrent prediction number
	rn = pred.shape[1] / 2

	loss_stack = np.array([])

	#range of valid comparison: 1:numsample
	num_sample = label.shape[0] - rn + 1

	#loss calculation
	for sample_ind in range(0, num_sample):
		loss_buf = 0
		for rn_ind in range(0, rn):
			loss_buf = loss_buf + np.linalg.norm(label[sample_ind+rn_ind,:] -
				pred[sample_ind, [rn_ind*2,rn_ind*2+1]])
			#print label[sample_ind+rn_ind,:], pred[sample_ind,[rn_ind*2, rn_ind*2+1]]

		loss_stack = np.hstack((loss_stack, loss_buf / rn))

	#sanity check
	assert(loss_stack.shape[0] == num_sample)

	#final loss
	#NOT CLEAR DIVIDE BY THE NUM_SAMPLE OR THE LENGTH OF THE TRAJECTORY
	return (np.sum(loss_stack) / num_sample)

def hdf5_read(h5_file_path):
	#read in test data
	with h5py.File(h5_file_path,'r') as f:
		h5_data_4D = f['data'][()]
		h5_label_2D = f['label'][()]

	num_testsample = h5_label_2D.shape[0]
	print "**************************************"
	print "Read %d test samples from %s" % (num_testsample, h5_file_path)
	return h5_data_4D, h5_label_2D

def cnn_predict(network_file_path, pretrained_model_path, h5_file_path, rn, pred_save_path=None):
	#check if the file exists
	if not os.path.isfile(pretrained_model_path and network_file_path and h5_file_path):
		print("Missing the input file")

	#initialize the caffe model
	net = caffe.Net(network_file_path, pretrained_model_path, caffe.TEST)

	#read in data
	h5_data_4D, h5_label_2D = hdf5_read(h5_file_path)

	#make sure the batch size and the saved hdf5 size are equal 
	assert(net.blobs['data'].data.shape == h5_data_4D.shape)

	#pre-assign the input blob
	net.blobs['data'].data[...] = h5_data_4D	
	input_blob_buffer = net.blobs['data'].data[...]
	#recurrent forecast
	pred_loc = np.array([])
	for rn_ind in range(0,rn):
		#forward pass the network
		net.forward()

		#collect the prediction
		loc = np.around(net.blobs['fc1'].data)

		if not pred_loc.any():
			pred_loc = loc
		else:
			pred_loc = np.hstack((pred_loc, loc))

		#update the input blob for recurrent prediction
		input_blob_prev = np.split(input_blob_buffer, [1], axis=3)[1]
		input_blob_new = np.split(input_blob_buffer, [1], axis=3)[0]
		input_blob_new = np.reshape(loc, input_blob_new.shape)
		input_blob = np.concatenate((input_blob_prev, input_blob_new), axis=3)
		net.blobs['data'].data[...] = input_blob
		input_blob_buffer = input_blob
	
	#save the predicted locations
	if pred_save_path:
		np.savetxt(pred_save_path, pred_loc, fmt='%d', delimiter=',')
		print "Generated %s" %(pred_save_path)

	#return the forecast loss
	return rc_loss(pred_loc, h5_label_2D)

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

def cnn_main(recurrent_depth, test_file_path):
	#use CPU mode
	caffe.set_mode_cpu()

	#train the caffe
	solver_file_path = './rc_solver.prototxt'
	#cnn_train(solver_file_path)

	#show the network infomation
	pretrained_model_path = '../model/_iter_10000.caffemodel'
	network_file_path = './rc_forecast.prototxt'
	#cnn_netinfo(network_file_path, pretrained_model_path)

	#perform forecast
	#pred_loc is Nx[x,y] (numpy array), where N is the number of test samples
	rn = 10 #set the recurrent number for prediction
	save_pred_to = '/home/hongjiw/research/data/RC/clips/pred_loc.txt'
	loss = cnn_predict(network_file_path, pretrained_model_path, test_file_path, rn, save_pred_to)
	print "Forecast loss with %d rccurrent prediction is: %f" %(rn, loss)
	
	#analysis	
	loss_stack = np.array([])
	for rn in range(1, recurrent_depth):
		loss = cnn_predict(network_file_path, pretrained_model_path, test_file_path, rn)
		loss_stack = np.hstack((loss_stack, loss))
		
	for rn in range(1, recurrent_depth):
		print "Loss with %d recurrent depth: %f" %(rn, loss_stack[rn-1])	
	
	return loss_stack


def static_main(recurrent_depth, test_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	h5_data_4D, h5_label_2D = hdf5_read(test_file_path)

	loss_stack = np.array([])
	for rn in range(1, recurrent_depth):
		pred_loc = np.reshape(h5_data_4D[:,:,:,h5_data_4D.shape[3]-1], h5_label_2D.shape)
		pred_loc = np.tile(pred_loc, [1, rn])
		loss = rc_loss(pred_loc, h5_label_2D)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, recurrent_depth):
		print "Loss with %d recurrent depth: %f" %(rn, loss_stack[rn-1])	

	return loss_stack

def constv_main(recurrent_depth, test_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	h5_data_4D, h5_label_2D = hdf5_read(test_file_path)

def consta_main(recurrent_depth, test_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	h5_data_4D, h5_label_2D = hdf5_read(test_file_path)


if __name__ == '__main__':
	recurrent_depth = 15;
	test_file_path = '/home/hongjiw/research/data/RC/clips/test.h5'
	#CNN MAIN FUNCTION
	#cnn_loss = cnn_main(recurrent_depth, test_file_path)
	#plt.plot(range(1,recurrent_depth), cnn_loss)

	#BASELINES
	#static_loss = static_main(recurrent_depth, test_file_path)
	#plt.plot(range(1,recurrent_depth), static_loss)

	constv_loss = constv_main(recurrent_depth, test_file_path)
	consta_loss = consta_main(recurrent_depth, test_file_path)


	#PLOT ALL MOTHODS
	"""
	plt.ylabel('Loss = Euc dis / Num frames')
	plt.xlabel('#Recurrent depth')
	plt.grid()
	pltself.show()
	"""