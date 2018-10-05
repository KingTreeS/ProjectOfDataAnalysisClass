# -*- encoding: UTF-8 -*-
"""
Author:KingTreeS
2018-10-1
Learn how to use decision tree
"""
from math import log
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle

"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calShannonEnt(dataSet):
    # 数据集的大小数量
    numberEntirs = len(dataSet)
    # 保存每个标签出现次数的字典
    labelCounts = {}
    # 提取标签数据并计数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 利用香农熵公式计算
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numberEntirs
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
函数说明:创建测试数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 特征标签
"""


def createDataSet():
    # 读取数据并处理
    file = open('/Volumes/D/GithubProject/ProjectOfDataAnalysisClass/决策树/DecisionTree/page-blocks.data', 'r')
    fileLines = file.readlines()
    fileLines[0] = fileLines[0].lstrip('\ufeff')
    hoRatio = 0.1
    numberLines = len(fileLines)
    m = int(hoRatio * numberLines)
    dataSet = []
    trainingDataSet = []

    for i in range(m):
        line = fileLines[i]
        line.strip()
        listFromLine = line.split()
        trainingDataSet.append(listFromLine)

    for j in range(m, numberLines):
        line = fileLines[j]
        line.strip()
        listFromLine = line.split()
        dataSet.append(listFromLine)
    """    
    for line in fileLines:
        line.strip()
        listFromLine = line.split()
        dataSet.append(listFromLine)
    """
    labels = ['HEIGHT', 'LENGTH', 'AREA', 'ECCEN', 'P_BLACK', 'P_AND', 'MEAN_TR', 'BLACKPIX', 'BLACKAND', 'WB_TRANS']
    return dataSet, labels, trainingDataSet


"""
函数说明:按照给定特征划分数据集
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # 循环判断当前特征值是否符合value
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去除当前特征值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


"""
函数说明:选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""


def chooseBestFeatureToSplit(dataSet):
    # 统计特征数量
    numberFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = calShannonEnt(dataSet)
    # 设定信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 循环遍历所有特征值
    for i in range(numberFeatures):
        # 记录每组特征值中的元素
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 设定经验信息熵
        newEntropy = 0.0
        # 循环遍历特征值上的元素计算出经验信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        # 计算增益信息熵
        infoGain = baseEntropy - newEntropy
        # 选取最大信息增益熵和对应的特征值索引
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
函数说明:统计classList中出现此处最多的元素(类标签)
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
函数说明:创建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""


def createTree(dataSet, labels):
    # 取分类标签
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征值时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最好的数据集划分方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 得到对应的标签值
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 删除已使用的特征标签
    subLabels = labels[:]
    del (subLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 递归调用创建决策树函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


"""
函数说明:获取决策树叶子结点的数目
Parameters:
    myTree - 决策树
Returns:
numLeafs - 决策树的叶子结点的数目
"""


def getNumLeafs(myTree):
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


"""
函数说明:获取决策树的层数
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


"""
函数说明:绘制结点
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"/System/Library/Fonts/STHeiti Medium.ttc", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


"""
函数说明:标注有向边属性值
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
"""


def plotMidText(cntrPt, parentPt, txtString):
    # 计算标注位置
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


"""
函数说明:绘制决策树
Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
"""


def plotTree(myTree, parentPt, nodeTxt):
    # 设置叶结点格式
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制子结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


"""
函数说明:创建绘制面板
Parameters:
    inTree - 决策树(字典)
Returns:
    无
"""


def createPlot(inTree):
    # 创建fig
    fig = plt.figure(1, facecolor='white')
    # 清空fig
    fig.clf()
    # 去掉x、y轴
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()


"""
函数说明:使用决策树分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""


def classify(inputTree, featLabels, testVec):
    # 获取决策树结点
    firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


"""
函数说明:存储决策树
Parameters:
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名
Returns:
    无
"""


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


"""
函数说明:读取决策树
Parameters:
    filename - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
"""


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels, trainingDataSet = createDataSet()
    myTree = createTree(dataSet, labels)

    for testVec in trainingDataSet:
        result = classify(myTree, labels, testVec)
        print(str(result))
    # print(myTree)
    # createPlot(myTree)
