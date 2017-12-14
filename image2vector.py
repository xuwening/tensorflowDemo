
import numpy as np

def image2vector(filename):
    vect = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            vect[0, 32*i+j] = int(line[j])
    return vect

