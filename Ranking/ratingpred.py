# Input: number of labels k
# vector theta of d rows, 1 column
# vector b of k-1 rows, 1 column
# vector x of d rows, 1 column
# Output: label (1 or 2 ... or k)
import numpy as np
def run(k,theta,b,x):
    # Your code goes here
    t=np.dot(np.squeeze(theta), np.squeeze(x))
    if(t<=b[0]): label=1
    for i in range(0, len(b)-1):
        if(b[i]<t and t<=b[i+1]): label=i+2
    if(t>b[len(b)-1]): label=len(b)+1
    return label
