"""
Author:KingTreeS
2018-10-1
识别文件中的数字
"""
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明：将文件中的像素转化为数据集矩阵
Parameters:
	filename - 读取的文件路径
Returns:
	returnMat - 返回的1*1024向量
"""
def imageToVector(filename):
	#读取文件
	file = open(filename)
	#创建默认值为0的1*1024向量
	returnMat = np.zeros((1,1024))
	#按行读取
	for i in range(32):
		lineStr = file.readline()
		#按每行的每列读取
		for j in range(32):
			returnMat[0,32*i+j] = lineStr[j]
	return returnMat

"""
函数说明：手写数字测试分类
Parameters:
	无
Returns:
	无
"""
def handwritingClassTest():
	#训练数据集的标签
	hwLabels = []
	#返回文件夹trainingDigits下的文件名
	trainingFileList = listdir("trainingDigits")
	#获得文件数
	m = len(trainingFileList)
	#创建默仍值为0的m*1024训练集矩阵
	trainingMat = np.zeros((m,1024))
	#依次对文件夹下的每个文件进行解析
	#获得文件名，文件标签和文件中的数据，构建训练数据集矩阵
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = imageToVector('trainingDigits/'+str(fileNameStr))
	#构建kNN分类器
	neigh = kNN(n_neighbors = 3,algorithm = 'auto')
	#拟合训练
	neigh.fit(trainingMat,hwLabels)
	#错误数检测
	errorCount = 0.0
	#读取testDigits文件夹下的文件，并且解析
	#利用predict进行预测返回分类结果
	testFileList = listdir('testDigits')
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		vectorUnderTest = imageToVector('testDigits/'+str(fileNameStr))
		classifierUnderTest = neigh.predict(vectorUnderTest)
		print("分类返回结果为："+str(classifierUnderTest)+"真实结果为："+str(classNumber))
		if classifierUnderTest != classNumber:
			errorCount += 1.0
	print("共算错了%d个数据\n错误率为%f%%" % (errorCount,errorCount/mTest * 100))

if __name__ == '__main__':
	handwritingClassTest()