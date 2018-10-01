# -*- coding: UTF-8 -*-
"""
Author:KingTreeS
2018-9-30
The KNN used in date
"""
import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import collections

"""
函数说明：读取文件并且解析数据
Parameters:
	filename - 记录约会数据的文件路径
Returns:
	returnMat - 训练数据集矩阵
	classLabelsVector - 训练数据集标签
"""
def fileToMatrix(filename):
	#打开对应的文件，并读取数据
	#file = open(filename,'r',encoding='UTF-8')
	file = open(filename,'r')
	arraylines = file.readlines()
	#对读取的数据进行处理
	#对于可能含有BOM的UTF-8文件，要去除BOM
	arraylines[0] = arraylines[0].lstrip('\ufeff')
	numberOfLines = len(arraylines)
	#以读取文件的行数和前三个字段为列数，创建默认值为0的数组
	returnMat = np.zeros((numberOfLines,3))
	#分类所需要的标签
	classLabelsVector = []
	index = 0

	for line in arraylines:
		#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
		#使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		#判断文本中的喜欢程度，分别由轻到重用1，2，3表示
		if listFromLine[-1] == 'didntLike':
			classLabelsVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelsVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelsVector.append(3)
		index += 1
	return returnMat,classLabelsVector

"""
函数说明：可视化数据
Parameteres：
	datingDataMat-训练数据集举证矩阵
	datingLabels-训练的数据标签
Returns:
	无
"""
def showDatas(datingDataMat,datingLabels):
	font = FontProperties(fname=r"/System/Library/Fonts/STHeiti Medium.ttc",size=14)
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
	fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))

	#为不同的标签分配颜色
	numberOfLines = len(datingLabels)
	LabelsColors = []
	for label in datingLabels:
		if label == 1:
			LabelsColors.append('black')
		elif label == 2:
			LabelsColors.append('blue')
		elif label == 3:
			LabelsColors.append('red')
	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=.5)
	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗的时间占比',FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	plt.setp(axs0_title_text,size=9,weight='bold',color='red')
	plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
	plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')

	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰淇淋)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5)
	#设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数',FontProperties=font)
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数',FontProperties=font)
	plt.setp(axs1_title_text,size=9,weight='bold',color='red')
	plt.setp(axs1_xlabel_text,size=7,weight='bold',color='black')
	plt.setp(axs1_ylabel_text,size=7,weight='bold',color='black')

	#画出散点图,以datingDataMat矩阵的第二列(玩游戏)、第三列（冰淇凌）数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5)
	#设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数',FontProperties=font)
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数',FontProperties=font)
	plt.setp(axs2_title_text,size=9,weight='bold',color='red')
	plt.setp(axs2_xlabel_text,size=7,weight='bold',color='black')
	plt.setp(axs2_ylabel_text,size=7,weight='bold',color='black')

	#设置图例
	didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
	smallDoses = mlines.Line2D([],[],color='blue',marker='.',markersize=6,label='smallDoses')
	largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])

	#显示图片
	plt.show()

"""
函数说明：对数据进行归一化
Parameters：
	dataSet - 训练数据集矩阵
Returns:
	normDataSet - 归一化的训练数据集矩阵
	ranges - 各项指标对应数据范围
	minVals - 各项指标对应最小值 
"""
def autoNorm(dataSet):
	#获得各项指标的最小和最大数据
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#获得各项指标的范围
	ranges = maxVals - minVals
	m = dataSet.shape[0]
	#利用shape返回矩阵行列数，创建默认值为0的归一化数据集矩阵
	normDataSet = np.zeros(np.shape(dataSet))
	#利用numpy中的tile（按照行和列分别重复某个列表）
	#对数据进行归一化处理，即用(dataSet-minVals)/ranges
	normDataSet = (dataSet - np.tile(minVals,(m,1)))/np.tile(ranges,(m,1))
	return normDataSet,ranges,minVals

"""
函数说明：通过训练数据集进行测试分类
Parameters:
	inx - 用于测试的数据
	group - 训练数据集矩阵
	labels - 训练数据集标签
	k - 选取的K位数
Returns:
	label - 分类后的标签
"""
def classify(inx,group,labels,k):
	#计算欧式距离
	distance = np.sum((inx-group)**2,axis=1)**0.5
	#对label进行排序
	label = [labels[index] for index in distance.argsort()[0:k]]
	#进行统计，获得数量最多的标签
	label = collections.Counter(label).most_common(1)[0][0]
	return label

"""
函数说明：分类起测试，判断正确率
Parameters:
	无
Returns:
	无
"""
def datingClassTest():
	#读取并解析数据
	filename = "dateTestSet.txt"
	returnMat,classLabelsVector = fileToMatrix(filename)
	#数据归一化处理
	normDataSet,ranges,minVals = autoNorm(returnMat)
	#选取数据集的前10%作为校验检测数据
	hoRatio = 0.10
	m = normDataSet.shape[0]
	numTestVecs = int(m*hoRatio)
	#错误分类计数
	errorCount = 0.0
	k = 4

	for i in range(numTestVecs):
		classifierResult = classify(normDataSet[i,:],normDataSet[numTestVecs:m,:],classLabelsVector[numTestVecs:m],k)
		print("分类结果是："+str(classifierResult)+"；真实结果是："+str(classLabelsVector[i]))
		if classifierResult != classLabelsVector[i]:
			errorCount += 1.0
	#计算错误率
	errors = (errorCount/float(numTestVecs))*100
	print("错误率为："+str(errors)+"%")

"""
函数说明：输入三个指标的数据，测试出喜欢程度
Parameters:
	无
Returns:
	无
"""
def classifyPerson():
	resultList = ['讨厌','有些喜欢','非常喜欢']
	#提示获得输入的三项指标
	gameTime = float(input('玩视频游戏所耗时间百分比:'))
	flyMiles = float(input('每年获得的飞行常客里程数:'))
	iceCream = float(input('每周消费的冰激淋公升数:'))
	inx = np.array([gameTime,flyMiles,iceCream])

	filename = "dateTestSet.txt"
	returnMat,classLabelsVector = fileToMatrix(filename)
	normDataSet,ranges,minVals = autoNorm(returnMat)
	k = 4
	#测试集也要归一化
	inx = (inx - minVals)/ranges
	classifierResult = classify(inx,normDataSet,classLabelsVector,k)
	print("你可能"+resultList[classifierResult-1]+"这个人。")

if __name__ == '__main__':
	#datingClassTest()
	classifyPerson()
