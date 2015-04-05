import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import shutil
import tempfile
import matplotlib.pyplot as plt
import h5py
import io

#set up the root path for caffe
caffe_root = '/home/hongjiw/research/library/caffe'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#check if the model file exists
import os


def rc_forecast(network_file_path, pretrained_model_path, h5_test_file):
	#check if the file exists
	if not os.path.isfile(pretrained_model_path and network_file_path and h5_test_file):
		print("Missing the input file")

	#initialize the caffe model
	net = caffe.Net(network_file_path, pretrained_model_path, caffe.TEST)
	
	#read in test data
	with h5py.File(h5_test_file,'r') as f:
		h5_data_4D = f['data'][()]
		h5_label_2D = f['label'][()]

	num_testsample = h5_label_2D.shape[0]
	print "Read %d test samples from %s" % (num_testsample, h5_test_file)

	#make sure the batch size and the saved hdf5 size are equal 
	assert(net.blobs['data'].data.shape == h5_data_4D.shape)

	#assign the input blob
	net.blobs['data'].data[...] = np.copy(h5_data_4D)

	#forward pass the network and get the loss (float)
	loss = net.forward().get('loss')
	loss = loss / num_testsample

	#extract the prediction and round it
	pred_loc = np.around(net.blobs['fc1'].data)

	return pred_loc, loss

def rc_train(solver_file_path):
	
	solver = caffe.get_solver(solver_file_path)
	solver.solve()
	solver.test_nets[0].forward()
	loss = solver.test_nets[0].blobs['loss'].data
	print "Test loss is: (Modify the code to enable batch test)"
	print loss

if __name__ == '__main__':

	#essential paths
	solver_file_path = './rc_solver.prototxt'
	network_file_path = './rc_train_test.prototxt'
	pretrained_model_path = '../model/_iter_10000.caffemodel'
	h5_test_file = '/home/hongjiw/research/data/RC/clips/test.h5'
	data_dir = '/home/hongjiw/research/data/RC/clips'

	#use the CPU
	caffe.set_mode_cpu()

	#train the caffe
	#rc_train(solver_file_path)

	#perform forecast
	#pred_loc is Nx[x,y] (numpy array), where N is the number of test samples
	pred_loc, loss = rc_forecast(network_file_path, pretrained_model_path, h5_test_file)

	#save to file and wrap up
	print "**************************************"
	print "Final loss: %f" %(loss)

	#save the predicted locations
	pred_loc_file_path = data_dir + '/pred_loc.txt'
	np.savetxt(pred_loc_file_path, pred_loc, fmt='%d', delimiter=',')
	print "Generated %s" %(pred_loc_file_path)
	#save the loss
	loss_file_path = data_dir + '/loss.txt'
	FILE = open(loss_file_path, 'w')
	FILE.write(str(loss))
	print "Generated %s" %(loss_file_path)
	print "**************************************"