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

def rc_loss(pred, label):
	#pred is a N x (2*rn) numpy matrix
	#label is a N x 2 numpy matrix
	#make sure input is valid
	#print pred.shape
	#print label.shape
	assert(not pred.shape[1] % 2)
	assert(not label.shape[1] % 2)
	assert(pred.shape[0] == label.shape[0])
	assert(pred.shape[1] <= label.shape[1])

	#rn: recurrent prediction number
	rn = pred.shape[1] / 2

	loss_stack = np.array([])

	#loss calculation
	for sample_ind in range(0, pred.shape[0]):
		loss_buf = 0
		for rn_ind in range(0, rn):
			loss_buf = loss_buf + np.linalg.norm(label[sample_ind, [rn_ind*2,rn_ind*2+1]] -
				pred[sample_ind, [rn_ind*2,rn_ind*2+1]])
		loss_stack = np.hstack((loss_stack, loss_buf / rn))
	return (np.sum(loss_stack) / loss_stack.shape[0])

def hdf5_read(file_path):
	#read in test data
	with h5py.File(file_path,'r') as f:
		data = f['data'][()]
		label = f['label'][()]
	num_test = label.shape[0]
	print "Read %d test samples from %s" % (num_test, file_path)
	return data, label

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

def diff(data):
	v_vec = np.array([]);
	count = 0
	for it in range(1, data.shape[3]):
		if(not v_vec.any()):
			v_vec = data[:,:,:,it] - data[:,:,:,it-1]
			v_vec = np.reshape(v_vec, np.hstack((v_vec.shape, np.array([1]))))
		else:
			v_vec_new = data[:,:,:,it] - data[:,:,:,it-1]
			v_vec_new = np.reshape(v_vec_new, np.hstack((v_vec_new.shape, np.array([1]))))
			v_vec = np.concatenate((v_vec, v_vec_new), axis=3)
	return v_vec

def diff_ave(data):
	v_vec = np.reshape(np.sum(diff(data), axis=3), [data.shape[0], data.shape[2]])
	v_vec /= data.shape[3] - 1
	return v_vec

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

def static_main(params, rc_path):
	if not os.path.isfile(rc_path['test_file']):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(rc_path['test_file'])

	loss_stack = np.array([])
	pred = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
	pred = np.tile(pred, [1, params['fc_depth']])
	loss = rc_loss(pred, label)

	print "Loss with %d forecast depth(static): %f" %(params['fc_depth'], loss)	

	loss_stack = np.array([])
	for rn in range(1, params['fc_depth']+1):
		pred = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
		pred = np.tile(pred, [1, rn])
		loss = rc_loss(pred, label)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, params['fc_depth']+1):
		print "Loss with %d forecast depth(static): %f" %(rn, loss_stack[rn-1])	

	return loss_stack

def constv_main(params, rc_path):
	if not os.path.isfile(rc_path['test_file']):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(rc_path['test_file'])

	#get average velocity
	v_ave = diff_ave(data)
	
	prev = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
	pred = np.zeros(v_ave.shape)

	loss_stack = np.array([])
	for rn in range(0, params['fc_depth']):

		if(rn == 0):
			pred = prev + v_ave
		else:
			pred = np.hstack((pred, pred[:,[pred.shape[1]-2, pred.shape[1]-1]] + v_ave))
		loss = rc_loss(pred, label)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, params['fc_depth']+1):
		print "Loss with %d forecast depth(const velocity): %f" %(rn, loss_stack[rn-1])

	return loss_stack

def consta_main(params, rc_path):
	if not os.path.isfile(rc_path['test_file']):
		print("Missing the input file")
	#read in test data
	data, label = hdf5_read(rc_path['test_file'])

	#make sure have enough dimention for acceleration calculation
	assert(data.shape[3] > 2)

	#get the last three frames
	data = data[:,:,:,range(data.shape[3]-3, data.shape[3])]	
	
	#get the velocity
	v_vec = diff(data)

	#get the last acceleration
	a_vec = diff(v_vec)
	a_vec = np.reshape(a_vec, [data.shape[0], data.shape[2]])

	v_last = np.reshape(v_vec[:,:,:,v_vec.shape[3]-1], [label.shape[0], 2])
	prev = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
	pred = np.zeros(v_last.shape)

	loss_stack = np.array([])
	for rn in range(0, params['fc_depth']):
		if(rn == 0):
			pred = prev + v_last + a_vec
		else:
			pred = np.hstack((pred, pred[:,[pred.shape[1]-2, pred.shape[1]-1]] + v_last + rn*a_vec))
		loss = rc_loss(pred, label)
		loss_stack = np.hstack((loss_stack, loss))

	for rn in range(1, params['fc_depth']+1):
		print "Loss with %d forecast depth(const acceleration): %f" %(rn, loss_stack[rn-1])	
	return loss_stack

if __name__ == '__main__':
	
	rc_path = {}
	params = {}
	
	rc_path['save_pred'] = '/home/hongjiw/research/data/RC/clips/pred_loc.txt'
	rc_path['test_file'] = '/home/hongjiw/research/data/RC/clips/test_motion_rec_10.h5'
	params['save'] = True
	params['retrain'] = True
	params['fc_depth'] = 10
	params['visual'] = False
	params['gpu'] = False

	#BASELINES
	#static movement
	static_loss_stack = static_main(params, rc_path)
	plt.plot(range(1, params['fc_depth']+1), static_loss_stack, label="static movement")
	#constant velocity movement
	constv_loss_stack = constv_main(params, rc_path)
	plt.plot(range(1, params['fc_depth']+1), constv_loss_stack, label="constant velocity")
	#constant acceleration movement
	consta_loss_stack = consta_main(params, rc_path)
	plt.plot(range(1, params['fc_depth']+1), consta_loss_stack, label="constant acceleration")
	
	#CNN 
	#rec 1 forecast (motion)
	rc_path['solver'] = './rc_solver_rec_1.prototxt'
	rc_path['model'] = '../model/rec_1_cfl_iter_10000.caffemodel'
	rc_path['network'] = './rc_forecast_rec_1.prototxt'
	cnn_loss_stack = cnn_main(params, rc_path)
	plt.plot(range(1, params['fc_depth']+1), cnn_loss_stack, label="cnn_rec_1_cfl")

	#rec 10 forecast (motion)
	rc_path['solver'] = './rc_solver_rec_10.prototxt'
	rc_path['model'] = '../model/rec_10_cfl_iter_10000.caffemodel'
	rc_path['network'] = './rc_forecast_rec_10.prototxt'
	
	params['fc_depth'] = 1
	#cnn_loss_stack = cnn_main(params, rc_path)
	#params['fc_depth'] = 10

	#plt.plot(params['fc_depth'], cnn_loss_stack, 'ro', label="cnn_rec_10_cfl")

	#CNN rec 1 forecast (OF)
	rc_path['test_file'] = '/home/hongjiw/research/data/RC/clips/test_OF_rec_10.h5'
	rc_path['solver'] = './rc_solver_rec_1_OF.prototxt'
	rc_path['model'] = '../model/rec_1_OF_cfl_iter_10000.caffemodel'
	rc_path['network'] = './rc_forecast_rec_1_OF.prototxt'

	params['retrain'] = False
	#cnn_loss_stack = cnn_main(params, rc_path)

	#rec 10 forecast (OF)
	rc_path['solver'] = './rc_solver_rec_10_OF.prototxt'
	rc_path['model'] = '../model/rec_10_OF_cfl_iter_10000.caffemodel'
	rc_path['network'] = './rc_forecast_rec_10_OF.prototxt'

	#PLOT ALL
	plt.ylabel('Loss')
	plt.xlabel('Forecast depth')
	plt.legend(loc=2)
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,1.2*x2,y1,y2))
	plt.grid()
	plt.show()