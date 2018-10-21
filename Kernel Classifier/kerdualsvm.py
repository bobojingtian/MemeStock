# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
import cvxopt as co
import numpy as np
import K

def run(X,y):
    # Your code goes here
    n=len(X)
    d=len(X[0])
    f=np.ones((n, 1))
    b=np.zeros((n, 1))
    A=np.zeros((n, n))
    H=np.zeros((n, n))
    for i in range(n):
        A[i][i]=-1
        f[i]=-1
    for i in range(n):
        for j in range(n):
            H[i][j]=y[i]*y[j]*K.run(X[i], X[j])
    alpha = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'), co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
    return alpha
