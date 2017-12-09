
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(x, dataset, labels, k):
    size = dataset.shape[0]
    diffMat = tile(x, (size, 1)) - dataset

    sqdiffMat = diffMat ** 2
    sqdistances = sqdiffMat.sum(axis=1)
    distances = sqdistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1

    print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def file2matrx(fileName):
    f = open(file)
    lines = f.readlines()
    numberOfLines = len(lines)
    mat = zeros((numberOfLines, 3))

    classLabelVector = []
    index = 0

    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        mat[index, :] = listFromLine[0:3]


if __name__ == '__main__':
    group, labels = createDataSet()
    print('result:', classify0([1, 0], group, labels, 3))