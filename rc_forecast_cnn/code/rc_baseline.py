import os
import numpy as np
from rc_io import *
from rc_metric import *

def diff(data):
	v_vec = np.array([]);
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
	v_vec /= (data.shape[3] - 1)
	return v_vec

def static_main(data, label, forecast_depth, logfile_path):

	#static prediction
	pred = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
	pred = np.tile(pred, [1, forecast_depth])
	
	#calculate the loss
	assert(pred.shape == label.shape)
	loss = rc_loss(pred, label)	

	#report the loss
	print 'static loss: ' + str(loss);

	#log this
	with open(logfile_path, "a") as logfile:
		logfile.write('static loss: ' + str(loss) + '\n')
	
def constv_main(data, label, forecast_depth, logfile_path):

	#get average velocity
	v_ave = diff_ave(data)
	
	prev = np.reshape(data[:,:,:,data.shape[3]-1], [label.shape[0], 2])
	pred = np.zeros(v_ave.shape)

	#constant velocity prediction
	for rn in range(0, forecast_depth):
		if(rn == 0):
			pred = prev + v_ave
		else:
			pred = np.hstack((pred, pred[:,[pred.shape[1]-2, pred.shape[1]-1]] + v_ave))
	
	#calculate the loss
	loss = rc_loss(pred, label)
		
	#report the loss
	print 'constv loss: ' + str(loss);

	#log this
	with open(logfile_path, "a") as logfile:
		logfile.write('constv loss: ' + str(loss) + '\n')

def consta_main(data, label, forecast_depth, logfile_path):

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

	#constant acceleration prediction
	for rn in range(0, forecast_depth):
		if(rn == 0):
			pred = prev + v_last + a_vec
		else:
			pred = np.hstack((pred, pred[:,[pred.shape[1]-2, pred.shape[1]-1]] + v_last + rn*a_vec))
	
	#calculate the loss
	loss = rc_loss(pred, label)

	#report the loss
	print 'consta loss: ' + str(loss);

	#log this
	with open(logfile_path, "a") as logfile:
		logfile.write('consta loss: ' + str(loss) + '\n')

def baseline_main(data, label, forecast_depth, logfile_path):

	#baselines
	static_main(data, label, forecast_depth, logfile_path)
	constv_main(data, label, forecast_depth, logfile_path)
	consta_main(data, label, forecast_depth, logfile_path)
