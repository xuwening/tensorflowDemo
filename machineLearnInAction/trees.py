from math import log
import operator
import copy


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannongEnt(dataSet):

    numEntries = len(dataSet)
    labelCounts = {}

    ## 统计yes和no标签各多少数据
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    ## 计算香浓值
    for key in labelCounts:
        ## 每个key所占总体比率
        prob = float(labelCounts[key]) / numEntries
        ## -(比率 * log2比率)
        ## 因为是负值，所以和就是0-
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

## 按轴分割数据第一列、第二列……，筛选第i列等于value的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplilt(dataSet):
    
    ## 列数
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannongEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        
        ## 当前列的集合
        featList = [example[i] for example in dataSet]
        ## 去重
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            ## 获取该特征分类后的集合
            subDataSet = splitDataSet(dataSet, i, value)
            ## 该集合占全集的比率
            prob = len(subDataSet) / float(len(dataSet))
            ## 比率 * 香浓值
            newEntropy += prob * calcShannongEnt(subDataSet)

        infoGain = baseEntropy - newEntropy

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0

        classCount[vote] += 1
    
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplilt(dataSet)

    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dat, labels = createDataSet()
    # print(chooseBestFeatureToSplilt(dat))
    tempLabels = copy.deepcopy(labels)
    myTree = createTree(dat, tempLabels)
    print(classify(myTree, labels, [1,1]))