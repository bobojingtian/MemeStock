# Input: number of labels k
# matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of labels, with n rows (samples), 1 column
# y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
# vector b of k-1 rows, 1 column
import numpy as np
import cvxopt as co
def run(k,X,y):
    # Your code goes here
    n=len(X)
    d=len(X[0])
    s=np.zeros((n, k - 1))
    for i in range(n):
        for l in range(k -1):
            if y[i]<=l+1: s[i][l]=-1
            else: s[i][l]=1
    H=np.zeros((d+k-1, d+k-1))
    for i in range(d): H[i][i]=1
    f=np.zeros((d+k-1, 1))
    A=np.zeros((n*(k-1)+k-2, d+k-1))
    for i in range(n*(k-1)):
        r=int(i/(k-1))
        c=i%(k-1)
        for j in range(d): A[i][j]=-s[r][c]*X[r][j]
        A[i][c+d]=s[r][c]
    c=d
    for i in range(n*(k-1), n*(k-1)+k-2):
        A[i][c]=1
        A[i][c+1]=-1
        c=c+1
    C=np.zeros((n*(k-1)+k-2, 1))
    for i in range(n*(k-1)): C[i]=-1
    z=np.array(co.solvers.qp(co.matrix(H,tc='d'), co.matrix(f, tc='d'), co.matrix(A,tc='d'), co.matrix(C, tc='d'))['x'])
    theta=np.zeros((d, 1))
    b=np.zeros((k-1, 1))
    for i in range(0, d): theta[i]=z[i]
    for i in range(d, d+k-1): b[i-d]=z[i]
    return (theta, b)
