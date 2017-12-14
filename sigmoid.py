
import sys
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        print(sigmoid(float(sys.argv[1])))
    else:
        print('please input arg x')