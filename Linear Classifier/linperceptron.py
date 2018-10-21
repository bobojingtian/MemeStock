# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sampleX = np.random.random((n,d))
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
import numpy as np

def run(L,X,y):
    # Your code goes here
    n=len(X)
    d=len(X[0])
    theta=np.zeros((1, d))
    theta[0]=0
    for i in range(0, L):
        for t in range(0, n):
            if(y[t]*np.dot(theta, X[t])<=0):
                for k in range(0, d):
                    theta[0][k]=theta[0][k]+y[t]*X[t][k]
    return theta.T

