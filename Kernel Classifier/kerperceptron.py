# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
import numpy as np
import K

def run(L,X,y):
	n = len(X)
	alpha = np.zeros(n)
	for i in range(0, L):
		for t in range(0, n):
			val = 0
			for x in range(0, n):
				val += alpha[x] * y[x] * K.run(X[x], X[t])
			if y[t] * val <= 0:
				alpha[t] += 1
	return alpha
