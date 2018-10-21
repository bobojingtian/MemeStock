# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
import numpy as np
import kerperceptron
import K

def run(alpha,X,y,z):
	s = 0
	n = len(X)
	for i in range(0,n):
		s += alpha[i]*y[i]*K.run(X[i],z)
	if(s>0):
		label = 1
	else:
		label = -1
	return label
