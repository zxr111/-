import numpy as np
import pandas as pd
import time

def loadData(filePath):
    print('开始读取数据')
    df = pd.read_csv(filePath)
    #第一个冒号代表索引，第二个代表每个向量对应的数据分片
    dataArr = df.iloc[:, 1:]
    dataLable = df.iloc[:, 0]

    return dataArr, dataLable

def calDist(x, x1):
    '''
    计算两点的欧氏距离
    :param x: 第一个点
    :param x1: 第二个点
    :return: 两点间的欧氏距离
    '''
    return np.sqrt(np.sum(np.square(x - x1)))

def getTopKLable(trainData, trainLable, x, topK):
    '''
    获取x在样本中最近的K个点分别对应相同标签的个数
    :param trainData: 样本数据
    :param trainLable: 样本标签
    :param x: 测试数据点
    :param topK: 最邻近的k个
    :return: 样本里测试点最近的K个点的标签对应个数
    '''
    trainDataMat = np.mat(trainData)
    trainLableMat = np.mat(trainLable).T
    #x和样本中所有点的距离，扩充为长度为所有样本集的长度
    dist = [0] * len(trainLableMat)
    #计算x和样本集中所有点的距离
    for i in range(len(trainDataMat)):
        xi = trainDataMat[i]
        xdist = calDist(x, xi)
        dist[i] = xdist
    #topKLable[0]个数为邻近0标签点的个数
    topKLable = [0] * topK
    #np.argsort返回排序后的索引位置，索引位置为该点在样本集中的位置
    topKIndex = np.argsort(np.array(dist))[:topK]
    #计算topk点对应的lable的计数
    for i in topKIndex:
        topKLable[int(trainLableMat[i])] += 1

    return topKLable

def test(trainData, trainLable, testData, testLable, topK=25, testSize=100):
    #将测试集转换为矩阵以便使用
    print('开始测试')
    testDataMat = np.mat(testData)
    testLableMat = np.mat(testLable).T
    errCnt = 0
    for i in range(testSize):
        print('共%d轮，正进行第%d轮' % (testSize, i))
        x = testDataMat[i]
        topKLable = getTopKLable(trainData, trainLable, x, topK)
        finalLable = np.argsort(np.array(topKLable))[-1]
        if finalLable != testLableMat[i]:
            errCnt += 1
    return 1 - (errCnt / testSize)

if __name__ == '__main__':
    trainData, trainLable = loadData('./Mnist/mnist_train/mnist_train.csv')
    testData, testLable = loadData('./Mnist/mnist_test/mnist_test.csv')
    start = time.time()
    print('准确率为：%f' % test(trainData, trainLable, testData, testLable))
    end = time.time()
    print('共花费时间为%ds' % (end - start))