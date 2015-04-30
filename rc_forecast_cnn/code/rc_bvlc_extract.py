import sys
import os
import numpy as np
import scipy.io as sio
import caffe
import cv2
import time
import Image
from misan import *
from rc_cnn import *

caffe_root = '/home/hongjiw/research/library/caffe'
sys.path.insert(0, caffe_root + 'python')

def bvlc_extract(rc_path, params):

	#load train, trainval, test list
	with open(rc_path['data_train_list']) as f:
		train_list = f.read().splitlines()

	with open(rc_path['data_trainval_list']) as f:
		trainval_list = f.read().splitlines()
	
	with open(rc_path['data_test_list']) as f:
		test_list = f.read().splitlines()

	#setup numpy arrays for train, trainval, test feature data
	train = {}
	trainval = {}
	test = {}

	#walk through the directories
	for dir in next(os.walk(rc_path['data_path']))[1]:

		#parse out the directory name and the tracker path file
		dir_path = rc_path['data_path'] + '/' + dir + '/' + 'imgs'
		tracker_path = rc_path['data_path'] + '/' + dir + '/tracker_loc.txt'
		
		#if the feature has been extracted, load the feature and return
		alex_feature_path = rc_path['data_path'] + '/' + dir + '/' + rc_path['alex_feature_filename']
		
		if not os.path.isfile(alex_feature_path):
			#otherwise, get a numpy array of image names
			img_names = np.array(next(os.walk(dir_path))[2])

			#the them in right order
			img_names = np.sort(img_names)

			#get a numpy array of tracker locations
			tracker_locs = np.loadtxt(tracker_path, delimiter=',')
			
			#something bad happen if the assert fails
			assert(img_names.shape[0] == tracker_locs.shape[0])
		
			#get a space to put all the features
			img_arr = np.zeros([img_names.shape[0], 3, params['resize'], params['resize']])
			
			#extract feature for all the images
			img_ind = 0
			for img_name in img_names:
				#parse to a single image file
				img_path = dir_path + '/' + img_name

				#read ine the image
				img = cv2.imread(img_path)
				
				#get the size of the image
				img_h, img_w = img.shape[0:2]

				#retreive the tracker location and the bounding box
				x, y = tracker_locs[img_ind, [0,1]]
				w, h = tracker_locs[img_ind, [2,3]]

				#round the box
				x = round(x)
				y = round(y)
				w = round(w)
				h = round(h)

				#get a larger range for feature extraction 
				center_x = x + w/2
				center_y = y + h/2
				w_offset = (w/2)*params['radius']
				h_offset = (h/2)*params['radius']

				x1 = (center_x - w_offset) if (center_x - w_offset) > 0 else 0
				y1 = (center_y - h_offset) if (center_y - h_offset) > 0 else 0
				x2 = (center_x + w_offset) if (center_x + w_offset) < img_w else img_w
				y2 = (center_y + h_offset) if (center_y + h_offset) < img_h else img_h

				x1 = round(x1)
				y1 = round(y1)
				x2 = round(x2)
				y2 = round(y2)

				#crop out the tracker loc from image
				img_crop = img[y1:y2, x1:x2]
				img_resize = {params['resize'], params['resize'], img_crop.shape[2]}
				img_crop = cv2.resize(img_crop, (params['resize'], params['resize']))

				#visualize for your own check
				#cv2.imshow("ff", img_crop)
				#cv2.waitKey(100)
				
				#get ready to be put into the space
				img_crop = np.swapaxes(img_crop, 1, 2)
				img_crop = np.swapaxes(img_crop, 0, 1)

				#put it into the space
				img_arr[img_ind,:,:,:] = img_crop
				img_ind = img_ind + 1

			#update the alex's network file
			template_params = {"input_num": img_arr.shape[0], "resize": params['resize']}
			update_template(rc_path['bvlc_test_template'], rc_path['bvlc_test'], template_params)

			#forward into alex's network
			#initialize the caffe model
			net = caffe.Net(rc_path['bvlc_test'], rc_path['bvlc_model'], caffe.TEST)
			
			#feedforward and return the feature of the last fully connected layer
			alex_feature = rc_cnn_predict(net, img_arr, 1, 'fc7')

			#save the extracted alex's feature as a mat file
			alex_feature_dic = {}
			alex_feature_dic['alex_feature'] = alex_feature
			sio.savemat(alex_feature_path, alex_feature_dic)

			print 'Saved ' + alex_feature_path

def rc_bvlc_extract(rc_path, params):

	#extract the alex's feature and save to mat
	bvlc_extract(rc_path, params)