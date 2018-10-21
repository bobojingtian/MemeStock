# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
import cvxopt as co
import numpy as np
from createsepdata import *
from linpred import *
from linperceptron import *

def run(X,y):
    # Your code goes here
    n=len(X)
    d=len(X[0])
    H=np.zeros((d, d))
    f=np.zeros(d)
    A=np.zeros((n, d))
    b=np.zeros(n)
    for i in range(0, d):
        H[i][i]=1
    for i in range(0, n):
        for j in range(0, d):
            A[i][j]=0-(y[i])*(X[i][j])
    for i in range(0, n):
        b[i]=-1
    theta = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'), co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
    return theta
