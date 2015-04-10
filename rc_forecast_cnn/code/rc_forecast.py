import numpy as np
import scipy as sp
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

def rc_loss(pred, label, test_seg):
	#pred is a N x (2*rn) numpy matrix
	#label is a N x 2 numpy matrix
	#make sure input is valid	
	assert(not pred.shape[1] % 2)
	assert(pred.shape[0] == label.shape[0] == np.sum(test_seg, axis=0))

	#rn: recurrent prediction number
	rn = pred.shape[1] / 2

	loss_stack = np.array([])

	#loss calculation
	offset = 0;
	for seg_num in test_seg:
		#range of valid comparison: 1:numsample
		num_sample = seg_num - rn + 1
		for sample_ind in range(0, num_sample):
			loss_buf = 0
			for rn_ind in range(0, rn):
				loss_buf = loss_buf + np.linalg.norm(label[offset+sample_ind+rn_ind,:] -
					pred[offset+sample_ind, [rn_ind*2,rn_ind*2+1]])
				#print label[sample_ind+rn_ind,:], pred[sample_ind,[rn_ind*2, rn_ind*2+1]]
			loss_stack = np.hstack((loss_stack, loss_buf / rn))

	#final loss
	return (np.sum(loss_stack) / loss_stack.shape[0])

def hdf5_read(file_path):
	#read in test data
	with h5py.File(file_path,'r') as f:
		data = f['data'][()]
		label = f['label'][()]
	num_test = label.shape[0]
	print "Read %d test samples from %s" % (num_test, file_path)
	return data, label

def rc_cnn_predict(net, data, rn, pred_save_path=None):
	#make sure the batch size and the saved hdf5 size are equal 
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
		loc = net.blobs['fc1'].data

		if not pred.any():
			pred = loc
		else:
			pred = np.hstack((pred, loc))

		#update the input blob for recurrent prediction
		input_blob_prev = np.split(input_blob_buffer, [1], axis=3)[1]
		input_blob_new = np.split(input_blob_buffer, [1], axis=3)[0]
		input_blob_new = np.reshape(loc, input_blob_new.shape)
		input_blob = np.concatenate((input_blob_prev, input_blob_new), axis=3)
		net.blobs['data'].data[...] = input_blob
		input_blob_buffer = input_blob

	#save the predicted locations
	if pred_save_path:
		np.savetxt(pred_save_path, pred, fmt='%d', delimiter=',')
		print "Prediction result saved to: %s" %(pred_save_path)

	#return the forecast loss
	return pred

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

def load_test_seg(testseg_file_path):
	test_seg = sp.io.loadmat(testseg_file_path)['test_seg']
	test_seg = np.array([np.asscalar(test_seg[0,0][0]), np.asscalar(test_seg[0,1][0])])
	return test_seg

def cnn_main(recurrent_depth, test_file_path, testseg_file_path):
	#use CPU mode
	caffe.set_mode_cpu()

	#train the caffe
	solver_file_path = './rc_solver.prototxt'
	#cnn_train(solver_file_path)

	#show the network infomation
	pretrained_model_path = '../model/_iter_10000.caffemodel'
	network_file_path = './rc_forecast.prototxt'
	#cnn_netinfo(network_file_path, pretrained_model_path)

	#initialize the caffe model
	net = caffe.Net(network_file_path, pretrained_model_path, caffe.TEST)
	
	#read in data
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	data, label = hdf5_read(test_file_path)

	#load test seg information
	test_seg = load_test_seg(testseg_file_path)
	#rc prediction
	rn = 10 #set the recurrent number for prediction
	save_pred_to = '/home/hongjiw/research/data/RC/clips/pred_loc.txt'
	pred = rc_cnn_predict(net, data, rn, save_pred_to)
	loss = rc_loss(pred, label, test_seg)
	print "Forecast loss with %d rccurrent prediction is: %f" %(rn, loss)
	
	#analysis	
	loss_stack = np.array([])
	for rn in range(1, recurrent_depth):
		pred = rc_cnn_predict(net, data, rn)
		loss = rc_loss(pred, label, test_seg)
		loss_stack = np.hstack((loss_stack, loss))
		
	for rn in range(1, recurrent_depth):
		print "Loss with %d recurrent depth: %f" %(rn, loss_stack[rn-1])	
	
	return loss_stack

def static_main(recurrent_depth, test_file_path, testseg_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(test_file_path)

	#load test seg information
	test_seg = load_test_seg(testseg_file_path)

	loss_stack = np.array([])
	for rn in range(1, recurrent_depth):
		pred = np.reshape(data[:,:,:,data.shape[3]-1], label.shape)
		pred = np.tile(pred, [1, rn])
		loss = rc_loss(pred, label, test_seg)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, recurrent_depth):
		print "Loss with %d recurrent depth: %f" %(rn, loss_stack[rn-1])	

	return loss_stack

def velocity_ave(data):
	v_vec = np.array([]);
	count = 0
	for it in range(1, data.shape[3]):
		if(not v_vec.any()):
			v_vec = data[:,:,:,it] - data[:,:,:,it-1]
		else:
			v_vec = v_vec + data[:,:,:,it] - data[:,:,:,it-1]

	v_vec = np.reshape(v_vec, [data.shape[0], data.shape[2]])
	v_vec /= data.shape[3] - 1
	return v_vec

def constv_main(recurrent_depth, test_file_path, testseg_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(test_file_path)

	#load test seg information
	test_seg = load_test_seg(testseg_file_path)

	#get average velocity
	v_ave = velocity_ave(data)
	#
	prev = np.reshape(data[:,:,:,data.shape[3]-1], label.shape)
	pred = np.zeros(v_ave.shape)

	loss_stack = np.array([])
	for rn in range(1, recurrent_depth):
		if(rn == 1):
			pred = prev + v_ave
		else:
			pred = np.hstack((pred, pred[:,[pred.shape[1]-2, pred.shape[1]-1]] + v_ave))
		loss = rc_loss(pred, label, test_seg)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, recurrent_depth):
		print "Loss with %d recurrent depth: %f" %(rn, loss_stack[rn-1])	

	return loss_stack

def consta_main(recurrent_depth, test_file_path):
	if not os.path.isfile(test_file_path):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(test_file_path)



if __name__ == '__main__':
	recurrent_depth = 5;
	test_file_path = '/home/hongjiw/research/data/RC/clips/test.h5'
	testseg_file_path = '/home/hongjiw/research/data/RC/clips/test_seg.mat'
	
	#CNN MAIN FUNCTION
	cnn_loss_stack = cnn_main(recurrent_depth, test_file_path, testseg_file_path)
	plt.plot(range(1,recurrent_depth), cnn_loss_stack, label="cnn")

	#BASELINES
	static_loss_stack = static_main(recurrent_depth, test_file_path, testseg_file_path)
	plt.plot(range(1,recurrent_depth), static_loss_stack, label="static movement")

	constv_loss_stack = constv_main(recurrent_depth, test_file_path, testseg_file_path)
	plt.plot(range(1,recurrent_depth), constv_loss_stack, label="constant velocity")

	#consta_loss = consta_main(recurrent_depth, test_file_path)

	#PLOT ALL MOTHODS
	
	plt.ylabel('Loss = Euc dis / Num frames')
	plt.xlabel('#Recurrent depth')
	plt.legend()
	plt.grid()
	plt.show()
	