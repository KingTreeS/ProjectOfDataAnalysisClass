# -*- coding: UTF-8 -*-
"""
Author:KingTreeS
2018-9-30
The simple of KNN
"""

import numpy as np
import operator
import collections

"""
定义训练的数据集和标签
"""
def createDataSet():
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	labels = ['投篮','投篮','灌篮','灌篮']
	return group,labels

"""
参数说明：
	inx-测试数据
	group-训练数据集
	labels-标签
	k-选择的距离最小的点个数
"""
def classify(inx,group,labels,k):
	#计算距离，欧式距离
	distance = np.sum((inx-group)**2,axis=1)**0.5
	#对计算的距离进行排序，并获得最小的k位标签
	k_labels = [labels[index] for index in distance.argsort()[0:k]]
	#对前k位标签进行统计，获得出现最多的标签值
	#排序之后获得数据形式为：列表包含元组
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	group,labels = createDataSet()
	inx = [30,110]
	#进行KNN聚类
	test_label = classify(inx,group,labels,3)
	print('当前选手的得分方式为：'+test_label)

