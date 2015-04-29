import h5py
def hdf5_read(file_path):
	#read in test data
	with h5py.File(file_path,'r') as f:
		data = f['data'][()]
		label = f['label'][()]
	num_test = label.shape[0]
	print "Read %d test samples from %s" % (num_test, file_path)
	return data, label