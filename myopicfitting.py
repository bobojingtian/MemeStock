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
    thetaS=np.zeros((F,1))
    tT=np.zeros((d,1))
    for f in range(0, F):
        z=np.zeros(n)
        tx=[]
        for j in S: tx.append(j)
        for t in range(0, n): z[t]=y[t]-np.dot(np.squeeze(tT[tx]), np.squeeze(X[t, tx]))
        DJ=np.zeros(d)
        for j in range(0, d):
            if j not in S:
                for t in range(0, n):
                    DJ[j]-=z[t]*X[t][j]
        jp=0
        tmpS=0
        for i in range(0, d):
            if(jp<abs(DJ[i]) and i not in S):
                jp=abs(DJ[i])
                tmpS=i
        tT[tmpS]=linreg.run(X[:, [tmpS]], z)
        S.append(tmpS)
    for i in range(0, F):
        thetaS[i]=tT[S[i]]
    return (S, thetaS)

