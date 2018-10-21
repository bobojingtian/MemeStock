# Input: numpy vector theta of d rows, 1 column
# numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
import numpy as np

def run(theta,x):
    # Your code goes here
    label=0
    z=np.dot(theta.T, x)
    if (z>0):
        label=1
    else:
        label=-1
    return label
