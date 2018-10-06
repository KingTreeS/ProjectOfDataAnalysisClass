# -*- encoding: UTF-8 -*-
"""
Author:KingTreeS
2018-10-1
Use KNN to recognize letters
"""
import numpy as np
import collections

"""
函数说明：读取文件信息，构建训练数据集
Parameters:
	filename - 存储数据的文件路径名
Return:
	returnMat - 训练数据集矩阵 
	labels - 训练数据集标签
"""
def fileToMat(filename):
	#从文件读取数据
	file = open(filename,'r')
	fileStr = file.readlines()
	#对数据进行处理
	#对于可能含有BOM的UTF-8文件，要去除BOM
	fileStr[0] = fileStr[0].lstrip('\ufeff')
	#以读取文件的行数和后16个字段为列数，创建默认值为0的数组
	numberOfLines = len(fileStr)
	returnMat = np.zeros((numberOfLines,16))
	#训练数据集所需标签
	labels = []
	index = 0

	for line in fileStr:
		#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
		listFromLine = line.split(',')
		#对returnMat相应的字段进行赋值
		returnMat[index,:] = listFromLine[1:]
		labels.append(listFromLine[0])
		index += 1
	return returnMat,labels

"""
函数说明：KNN算法，分类器
Parameters:
	inX - 用于分类的数据（测试集）
	dataSet - 用于训练的数据（训练集）
	labels - 分类标签
	k - KNN算法参数，选择距离最小的K个点
Returns:
	sortedLabel - 分类结果
"""
def classify(inX,dataSet,labels,k):
	#通过欧式距离求解
	distance = np.sum((inX - dataSet)**2,axis=1)**0.5
	#对计算的距离进行排序，并获得最小的k位标签
	k_labels = [labels[index] for index in distance.argsort()[0:k]]
	#对前k位标签进行统计，获得出现最多的标签值
	#排序之后获得数据形式为：列表包含元组
	sortedLabel = collections.Counter(k_labels).most_common(1)[0][0]
	return sortedLabel

"""
函数说明：对训练数据集进行测试
Parameters:
	无
Returns:
	无
"""
def letterClassTest():
	returnMat,labels = fileToMat('/Volumes/D/GithubProject/ProjectOfDataAnalysisClass/第二周/KNN/letter-recognition.data')
	#设置提取训练集的百分比作为测试数据
	m = len(returnMat)
	hoRatio = 0.05
	numberTest = int(m * hoRatio)
	#构建训练数据集和训练数据标签
	trainingDataSet = returnMat[numberTest:m,:]
	trainingLabels = labels[numberTest:m]
	#设置错误计数器
	errorCount = 0.0
	#以此分类并计算错误率
	for i in range(numberTest):
		classifierResult = classify(returnMat[i:i+1,:],trainingDataSet,trainingLabels,4)
		print("分类返回结果为："+str(classifierResult)+";真实结果为："+str(labels[i]))
		if classifierResult != labels[i]:
			errorCount += 1.0
	print("分类出错数为：%d;\n分类出错率为：%f%%" % (errorCount,errorCount/numberTest * 100))

if __name__ == '__main__':
	letterClassTest()
	
