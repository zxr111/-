
import numpy as np
import time
import pandas as pd

def loadData(filePath):
    '''
    读取Mnist训练数据集
    :param filePath: Mnist训练数据集的路径
    :return: 数据集数组dataArr和数据标记数组dataLable
    '''
    print('开始读取数据')
    df = pd.read_csv(filePath, header=None)
    dataArr = df.iloc[:, 1:]
    dataLable = df.iloc[:, 0]
    #线性分割只有两个类别 1 和 -1
    for i in range(len(dataLable)):
        if dataLable[i] >= 5:
            dataLable[i] = 1
        else:
            dataLable[i] = -1
    # for k in range(len(dataArr)):
    #     for i in range(len(dataArr[0])):
    #         print(dataArr[k][i])
    # f = open(filePath, 'r')
    # for line in f.readlines():
    #     #strip()用于去除 '/n' 再用 ','分割成数组
    #     curLine = line.strip().split(',')
    #     #线性分割只有两个类别 1 和 -1
    #     if int(curLine[0]) >= 5:
    #         dataLable.append(1)
    #     else:
    #         dataLable.append(-1)
    #     #加载数据集并进行归一化处理
    #     dataArr.append([int(num)/255 for num in curLine[1:]])
    return dataArr, dataLable

def train(dataArr, dataLable, iter=50, r=0.001):
    '''
    训练感知机
    :param filePath: 训练文件路径
    :param iter: 迭代次数，默认为50
    :param r: 学习率 默认为0.001
    :param w: 需要训练的超平面线性参数 默认初始值为0
    :param b: 需要训练的超平面截距 默认初始值为0
    :return: 训练好的 w, b
    '''

    #把读入的数据转化为数组形式
    trainData = np.mat(dataArr)
    #把读入标签转化为 N * 1形式
    trainLable = np.mat(dataLable).T
    #读取训练数据数组大小
    m, n = np.shape(trainData)
    #初始化w, b。w为n维横向量，与原向量长度保持一致
    w = np.zeros((1, n))
    b = 0
    #开始进行迭代训练
    #寻找损失函数的极值点
    print('开始训练')
    for k in range(0, iter):
        for i in range(0, m):
            xi = trainData[i]
            yi = trainLable[i]
            #损失函数大于0说明数据该点不符合
            #其中 w * xi + b 为函数距离
            # 当函数距离大于0，该点在超平面上方
            # yi应为1再乘以-1则损失函数小于0。反之相同
            # 故损失函数大于0时不符合，要进行梯度调整
            # 等于0时在超平面上也不符合
            # w的梯度为，对w求导 y1 * xi
            # b的梯度为，对b求导 yi
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + r * yi * xi
                b = b + r * yi

        print('第%d轮训练结束共%d轮' % (k+1, iter))

    return w, b

def test(testArr, testLable, w, b):
    print('开始测试')
    #模型预测错误个数
    errCnt = 0
    testMat = np.mat(testArr)
    testLable = np.mat(testLable).T
    m, n = np.shape(testArr)

    for i in range(m):
        xi = testMat[i]
        yi = testLable[i]
        if -1 * yi * (w * xi.T + b) >= 0:
            errCnt += 1
    #返回正确率
    return 1 - (errCnt / m)

if __name__ == '__main__':
    dataArr, dataLable = loadData('./Mnist/mnist_train/mnist_train.csv')
    testArr, testLable = loadData('./Mnist/mnist_test/mnist_test.csv')
    start = time.time()
    w, b = train(dataArr, dataLable, 10)
    print(w, b)
    end = time.time()
    print('训练时间为：%ds' % (end - start))
    accruRate = test(testArr, testLable, w, b)
    print('准确率为：%f' % accruRate)