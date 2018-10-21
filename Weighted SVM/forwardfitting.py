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
def run(F,X,y):
    n=len(X)
    d=len(X[0])
    S=[]
    thetaS=np.zeros((F, 1))
    tT=np.zeros((d, 1))
    for f in range(0, F):
        J=np.ones((d,1))
        J=-J
        z=np.zeros(n)
        tx=[]
        for i in S: tx.append(i)
        for t in range(0, n): z[t]=y[t]-np.dot(np.squeeze(tT[tx]), np.squeeze(X[t, tx]))
        for j in range(0, d):
            if j not in S: tT[j]=linreg.run(X[:,[j]], z)
            for t in range(0, n): J[j]+=(z[t]-np.dot(tT[j], X[t, j]))**2/2
        jp=100000
        tmpS=0
        for i in range(0, d):
            if(J[i]>0 and jp>J[i]):
                jp=J[i]
                tmpS=i
        S.append(tmpS)
    for i in range(0, F):
        thetaS[i]=tT[S[i]]
    return (S, thetaS)
