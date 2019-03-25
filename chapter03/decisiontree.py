# 决策树的一般流程
from math import log
import operator

#　计算香农信息熵
def calcShannonEnt(dataSet):
    # 计算实例总数
    numEtries = len(dataSet)
    # 为所有分类创建字典
    labelConuts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelConuts.keys():
            labelConuts[currentLabel] = 0
        labelConuts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底求对数
    for key in  labelConuts:
        prob = float(labelConuts[key])/numEtries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照給定特征劃分數據集
# dataSet帶劃分數據集
# axis劃分數據集的特征
# 需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 從數據集中選取最好的特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    # 创建键值为classList中唯一值的数据字典,字典对象存储了每个标签出现的次数
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 操作键值排序字典,并返回出现次数最多的分类名称
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables)
    return myTree


##获取节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    #firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]#找到输入的第一个元素
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 1
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]#找到输入的第一个元素
    #firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
    if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


def classify(inputTree, featLabels, testVec):
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# 接下来使用Matplotlib注解绘制树形图
# 硬盘上存储决策树
# 递归构建决策树,采用递归的原则处理数据集
def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


myDataset, lables = createDataset()
print(lables)
myTree = retrieveTree(0)
print(myTree)
print(classify(myTree, lables, [1, 0]))
print(classify(myTree, lables, [1, 1]))
# print(myDataset)
# print(lables)
# shannonEnt = calcShannonEnt(myDataset)
# print(shannonEnt)
# myDataset[0][-1] = 'maybe'
# print(calcShannonEnt(myDataset))

# data = splitDataSet(myDataset, 0, 1)
# print(data)
#
# data1 = splitDataSet(myDataset, 0, 0)
# print(data1)
print(chooseBestFeatureToSplit(myDataset))