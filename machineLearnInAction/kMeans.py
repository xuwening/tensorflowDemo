
from numpy import *
import os


def loadDataSet(fileName):
    return loadtxt(fileName)
    # dataMat = []
    # f = open(fileName)

    # for line in f.readlines():
    #     curLine = line.strip().split('\t')
    #     fltLine = map(float, curLine) ##转换数据类型为float
    #     dataMat.append(list(fltLine))

    # return dataMat

def distEclud(vecA, vecB):
    ## 两向量的标准差
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]  ## 获取列数
    centroids = mat(zeros((k, n)))  ## 构建k行n列

    for j in range(n):
        print(dataSet[:, j])
        minJ = min(dataSet[:, j]) ##J列的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ) ## j列所有值与最小值差
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)

    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]

    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)

        return centroids, clusterAssment


if __name__ == '__main__':
    
    datMat = loadDataSet('testSet.txt')
    myCentroids, clustAssing = kMeans(datMat,4)