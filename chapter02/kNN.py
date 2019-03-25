from numpy import *
import numpy as np
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def createdataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# inX 用于分类的输入向量
# dataSet 训练的样本集
# labels 标签向量
# k表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    # 每年获得的飞行常客里程数
    # 玩视频游戏锁好的时间百分比
    # 每周消费的冰激凌公升数
    fr = open(filename)
    arrayOflines = fr.readlines()
    # 获取文件行数
    numbersOfLines = len(arrayOflines)
    # 创建以0填充的矩阵numPy
    matrix = zeros((numbersOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        # 截取掉所有的回车符
        line = line.strip()
        # 将整行数据分割成元素列表
        listFromLine = line.split('\t')
        # 选取前3个元素,存储到特征矩阵中
        matrix[index, :] = listFromLine[0: 3]
        # 将最后一列存储到向量classLabelVector中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return matrix, classLabelVector


def autoNorm(dataset):
    # 参数0使得函数可以从列中选取最小值
    minValue = dataset.min(0)
    # 参数0使得函数可以从列中选取最大值
    maxValue = dataset.max(0)

    ranges = maxValue - minValue
    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0]
    # tile(A, b) 将数组A重复b次
    normDataSet = dataset - tile(minValue, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minValue


def datingClassTest():
    hoRtio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normat, ranges, minvalues = autoNorm(datingDataMat)
    m = normat.shape[0]
    numTestVecs = int(m * hoRtio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normat[i, :], normat[numTestVecs:m, :],
                                     datingLabels[numTestVecs: m], 3)
        print("the classfier came back with : %d, the real answer is : %d " %(classifierResult, datingLabels[i]))
        if(classifierResult!= datingLabels[i]):
            errorCount += 1
    print("the total error rate is : %f " % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("play video games?"))
    ffMiles = float(input("miles?"))
    iceCream = float(input("iceCream?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normat, ranges, minvalues = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResult = classify0((inArr - minvalues)/ranges, normat, datingLabels, 3)
    print('you will probably like this person : ', resultList[classfierResult - 1])


def img2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        # 输入用于分类的inX, 输入训练样本集为ｄａｔａＳｅｔ
        # 标签向量，ｋ选择的最近邻居的数目。
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classfier came back with : %d, the real answer is : %d " %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1
    print('\n the total number of errors is : %d' %errorCount)
    print('\n the total error rate is: %f' %(errorCount / float(mTest)))


testVector = img2Vector('testDigits/0_13.txt')
handwritingClassTest()
# print(testVector[0, 0:31])
# print(testVector[0, 32:63])

# classifyPerson()
# datingClassTest()

# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# normat, ranges, minvalues = autoNorm(datingDataMat)
# print(normat)
# print(ranges)
# print(minvalues)

# print(datingDataMat)
# print(datingLabels[0:20])

# fig = plt.figure()
# ax = fig.add_subplot(111)
# 使用散点图显示第二列/第三列数据
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
#            15.0*array(datingLabels), 15.0*array(datingLabels))

# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#            15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()
