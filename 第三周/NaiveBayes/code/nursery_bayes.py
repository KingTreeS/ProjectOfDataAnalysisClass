# -*- encoding: UTF-8 -*-
"""
Author:KingTreeS
2018-10-8
Use naive bayes to analyse nursery database
"""

import numpy as np

"""
函数说明：读取文件，创建实验样本
Parameters:
    filename - 读取的文件路径
Returns:
    trainingDataSet - 用于训练的数据集
    labels - 用于训练的数据标签
    testDataSet - 用于测试的数据集
"""


def loadDataSet(filename):
    # 读取文件并进行处理
    fp = open(filename, 'r')
    fileLines = fp.readlines()
    fileLines[0] = fileLines[0].lstrip('\ufeff')
    fileDataSet = []
    labels = []

    for line in fileLines:
        line = line.strip()
        listLineData = line.split(',')
        fileDataSet.append(listLineData[:])
        labels.append(listLineData[-1])

    # 取获得数据的前12000条作为训练数据，剩余条数作为测试数据
    trainingDataSet = fileDataSet[:12000][:-1]
    testDataSet = fileDataSet[12000:-1]

    return trainingDataSet, labels, testDataSet


"""
函数说明：处理训练数据集，获得不重复出现的关键词
Parameters:
    trainingDataSet - 训练数据集
Returns:
    vocabDataSet - 无重复关键词的列表
"""


def createVocabList(trainingDataSet):
    vocabDataSet = set([])

    # 取并集
    for data in trainingDataSet:
        vocabDataSet = vocabDataSet | set(data)

    return list(vocabDataSet)


"""
函数说明：根据关键词表，将输入的数据向量化，为0或1
Parameters:
    vocabDataSet - 关键词列表
    inputSet - 需要向量化的数据
Returns:
    returnVec - 向量化后的列表
"""


def setOfWordsToVec(vocabDataSet, inputSet):
    returnVec = [0] * len(vocabDataSet)

    # 循环获取inputSet中的单词并判断是否在关键词列表中
    for word in inputSet:
        if word in vocabDataSet:
            returnVec[vocabDataSet.index(word)] = 1
        else:
            print(word + " is not in this Vocabulary")

    return returnVec


"""
函数说明：朴素贝叶斯分类器训练
Parameters:
    trainDataMat - 训练集向量化后的矩阵
    labels - 训练集的标签
Returns:
    pNotRecom - 文档属于not_recom的概率
    pRecom - 文档属于recommend的概率
    pVeryRecom - 文档属于very_recom的概率
    pPrior - 文档属于priority的概率
    pSpecPrior - 文档属于spec_priority的概率
    pNotRecomVec - not_recom的条件概率数组
    pRecomVec - recommend的条件概率数组
    pVeryRecomVec - very_recom的条件概率数组
    pPriorVec - priority的条件概率数组
    pSpecPriorVec - spec_prior的条件概率数组
"""


def trainNavBayes(trainDataMat, labels):
    # 获得训练集的数目以词条数
    numOfTrainDoc = len(trainDataMat)
    numOfWords = len(trainDataMat[0])
    # 统计属于各类别的概率
    pNotRecom = labels.count('not_recom')/float(numOfTrainDoc)
    pRecom = labels.count('recommend')/float(numOfTrainDoc)
    pVeryRecom = labels.count('very_recom')/float(numOfTrainDoc)
    pPrior = labels.count('priority')/float(numOfTrainDoc)
    pSpecPrior = labels.count('spec_prior')/float(numOfTrainDoc)
    # 初始化默认值为1的数组，拉普拉斯平滑
    pNotRecomNum = np.ones(numOfWords)
    pRecomNum = np.ones(numOfWords)
    pVeryRecomNum = np.ones(numOfWords)
    pPriorNum = np.ones(numOfWords)
    pSpecPriorNum = np.ones(numOfWords)
    # 初始化分母，设为2.0
    pNotRecomDenom = 2.0
    pRecomDenom = 2.0
    pVeryRecomDenom = 2.0
    pPriorDenom = 2.0
    pSpecPriorDenom = 2.0

    for i in range(numOfTrainDoc):
        if labels[i] == 'not_recom':
            pNotRecomNum += trainDataMat[i]
            pNotRecomDenom += sum(trainDataMat[i])
        elif labels[i] == 'recommend':
            pRecomNum += trainDataMat[i]
            pRecomDenom += sum(trainDataMat[i])
        elif labels[i] == 'very_recom':
            pVeryRecomNum += trainDataMat[i]
            pVeryRecomDenom += sum(trainDataMat[i])
        elif labels[i] == 'priority':
            pPriorNum += trainDataMat[i]
            pPriorDenom += sum(trainDataMat[i])
        elif labels[i] == 'spec_prior':
            pSpecPriorNum += trainDataMat[i]
            pSpecPriorDenom += sum(trainDataMat[i])
    # 计算属于各类别的条件概率值
    pNotRecomVec = np.log(pNotRecomNum / pNotRecomDenom)
    pRecomVec = np.log(pRecomNum / pRecomDenom)
    pVeryRecomVec = np.log(pVeryRecomNum / pVeryRecomDenom)
    pPriorVec = np.log(pPriorNum / pPriorDenom)
    pSpecPriorVec = np.log(pSpecPriorNum / pSpecPriorDenom)

    return pNotRecom, pRecom, pVeryRecom, pPrior, pSpecPrior, pNotRecomVec, pRecomVec, pVeryRecomVec, pPriorVec, pSpecPriorVec


"""
函数说明：朴素贝叶斯分类器的分类函数
Parameters:
    testDataMat - 测试集向量化后的矩阵
    pNotRecom - 文档属于not_recom的概率
    pRecom - 文档属于recommend的概率
    pVeryRecom - 文档属于very_recom的概率
    pPrior - 文档属于priority的概率
    pSpecPrior - 文档属于spec_priority的概率
    pNotRecomVec - not_recom的条件概率数组
    pRecomVec - recommend的条件概率数组
    pVeryRecomVec - very_recom的条件概率数组
    pPriorVec - priority的条件概率数组
    pSpecPriorVec - spec_prior的条件概率数组
Returns:
    classifyResult - 分类后的结果矩阵
"""


def classifyNavBayes(testDataMat, pNotRecom, pRecom, pVeryRecom, pPrior, pSpecPrior, pNotRecomVec, pRecomVec, pVeryRecomVec, pPriorVec, pSpecPriorVec):
    # 初始化分类后结果矩阵
    classifyResult = []

    for testData in testDataMat:
        # 从左到右对一个序列的项累计地应用有两个参数的函数，以此合并序列到一个单一值，防止下溢出
        pClassifyNotRecom = sum(testData * pNotRecomVec) + np.log(pNotRecom)
        pClassifyRecom = sum(testData * pRecomVec) + np.log(pRecom)
        pClassifyVeryRecom = sum(testData * pVeryRecomVec) + np.log(pVeryRecom)
        pClassifyPrior = sum(testData * pPriorVec) + np.log(pPrior)
        pClassifySpecPrior = sum(testData * pSpecPriorVec) + np.log(pSpecPrior)
        # 选择最大的概率值
        maxClassify = sorted([pClassifyNotRecom, pClassifyRecom, pClassifyVeryRecom, pClassifyPrior, pClassifySpecPrior])[-1]
        # 判断最大概率值所属类别
        if maxClassify == pClassifyNotRecom:
            finalResult = 'not_recom'
        elif maxClassify == pClassifyRecom:
            finalResult = 'recommend'
        elif maxClassify == pClassifyVeryRecom:
            finalResult = 'very_recom'
        elif maxClassify == pClassifyPrior:
            finalResult = 'priority'
        elif maxClassify == pClassifySpecPrior:
            finalResult = 'spec_prior'
        # 添加返回结果
        classifyResult.append([pClassifyNotRecom, pClassifyRecom, pClassifyVeryRecom, pClassifyPrior, pClassifySpecPrior, finalResult])

    return classifyResult


"""
函数说明：测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
"""


def testNavBayes():
    # 初始化各项数据
    filename = '/Volumes/D/软微课程作业/数据分析工具实践/第三周/1801220013-束超哲-第三次作业/NaiveBayes/data/nursery.data'
    trainingDataSet, labels, testDataSet = loadDataSet(filename)
    vocabDataSet = createVocabList(trainingDataSet)
    trainDataMat = []
    testDataMat = []
    # 分别获得向量化后的训练和测试矩阵
    for trainData in trainingDataSet:
        trainDataMat.append(setOfWordsToVec(vocabDataSet, trainData))
    for testData in testDataSet[:-1]:
        testDataMat.append(setOfWordsToVec(vocabDataSet, testData))
    # 训练朴素贝叶斯分类器
    pNotRecom, pRecom, pVeryRecom, pPrior, pSpecPrior, pNotRecomVec, pRecomVec, pVeryRecomVec, pPriorVec, pSpecPriorVec = trainNavBayes(trainDataMat, labels)
    # 获得分类结果
    classifyResult = classifyNavBayes(testDataMat, pNotRecom, pRecom, pVeryRecom, pPrior, pSpecPrior, pNotRecomVec, pRecomVec, pVeryRecomVec, pPriorVec, pSpecPriorVec)
    # 分类错误数目统计
    errorCount = 0
    for i in range(len(testDataMat)):
        finalResult = classifyResult[i][-1]
        print("pClassifyNotRecom:", classifyResult[i][0])
        print("pClassifyRecom:", classifyResult[i][1])
        print("pClassifyVeryRecom:", classifyResult[i][2])
        print("pClassifyPrior:", classifyResult[i][3])
        print("pClassifySpecPrior:", classifyResult[i][4])
        print(str(testDataSet[i]) + " 属于 " + str(finalResult))

        if finalResult != testDataSet[i][-1]:
            errorCount += 1

    error = float(errorCount / len(testDataMat)) * 100
    print("分类错误数为：", errorCount)
    print("分类错误率为：", error)


if __name__ == '__main__':
    testNavBayes()