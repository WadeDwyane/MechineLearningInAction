from numpy import *
import re

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 0 无侮辱;1 侮辱
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 获取词汇表,不会出现重复的词汇
def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 分别表示词汇表中的单词在输入文档中是否存在
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word : %s is not in my Vocabulary!' % word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0
    p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive


def bagOfWords2VecMN(vecabList, inputSet):
    returnVec = [0]*len(vecabList)
    for word in inputSet:
        if word in vecabList:
            returnVec[vecabList.index(word)] += 1
    return returnVec


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))


mySent = 'this book is the best book on Python or M.L. I have ever laid eyes upon.'
# print(mySent.split())
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
print(listOfTokens)
print([tok for tok in listOfTokens if len(tok) > 0])
# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(p0V)
# print(p1V)
# print(pAb)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOPosts[0]))