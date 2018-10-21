# Input: number of features F
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Output: numpy vector of selected features S, with F rows, 1 column
# numpy vector thetaS, with F rows, 1 column
# thetaS[0] corresponds to the weight of feature S[0]
# thetaS[1] corresponds to the weight of feature S[1]
# and so on and so forth
import numpy as np
import linreg
def SUM(X, y, theta):
    J=0
    for t in range(len(X)): J+=(y[t, 0]-np.dot(X[t], theta)[0])**2
    return J/2
def run(F,X,y):
    n=len(X)
    d=len(X[0])
    S=[]
    for f in range(1, F+1):
        MP=dict()
        for j in range(d):
            if j not in S:
                tmpS=S+[j]
                thetaS=linreg.run(X[:, tmpS], y)
                J=SUM(X[:, tmpS], y, thetaS)
                MP.setdefault(J, j)
        S+=[MP[min(MP.keys())]]
    return (np.asarray(S).reshape((F, 1)), linreg.run(X[:, S], y))