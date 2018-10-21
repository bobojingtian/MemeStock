# Input: number of iterations L
# number of labels k
# matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of labels, with n rows (samples), 1 column
# y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
# vector b of k-1 rows, 1 column
import numpy as np
def run(L,k,X,y):
    # Your code goes here
    n=len(X)
    d=len(X[0])
    theta=np.zeros((d, 1))
    b=np.zeros((k-1, 1))
    s=np.zeros((n, k-1))
    for i in range(0, k-1): b[i]=i
    for t in range(0, n):
        for l in range(0, k-1):
            if(y[t]<=l+1): s[t][l]=-1
            else: s[t][l]=1
    for i in range(0, L):
        for t in range(0, n):
            E=set()
            SV=0
            for l in range(k-1):
                if(s[t][l]*(np.vdot(theta.reshape(d,1), X[t,:].reshape(1,d))-b[l])<=0):
                    E.add(l)
                    SV+=s[t][l]
            if(len(E)>0):
                theta+=SV*X[t,:].reshape(d,1)
                for l in E:
                    b[l]=b[l]-s[t][l]
    return (theta, b)
