import numpy as np
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
