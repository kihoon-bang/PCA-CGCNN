import numpy as N
import numpy as np
import numpy.matlib
from numpy import linalg as LA



def pca_fnc(data):

	dataSize = N.shape(data)

	# Subtract off the mean for each dimension
	mn = data.mean(axis=1)
	mn = np.array([mn])
	mn = mn.T

	mn_rep = np.matlib.repmat(mn,1,dataSize[1])
	

	a = numpy.matrix(data)
	b = numpy.matrix(mn_rep)

	data = a - b


	covariance = 1.0/float(dataSize[1]-1)*data*data.T
	

	v, PC = LA.eig(covariance)


	rindices = (-v).argsort()[:]
	

	v = v[rindices]


	PC = PC[:,rindices]
	PC = PC.real

	signals = np.dot(PC.T,data)


	return signals, PC, v,mn
	
