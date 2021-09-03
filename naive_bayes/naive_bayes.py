import numpy as np
import time

def load_data(filename):
    data_arr = []
    data_lable = []

    print('开始读取数据')
    f = open(filename)
    for line in f.readlines():
        cur_line = line.strip().split(',')
        data_lable.append(int(cur_line[0]))
        #像素点大于128的设置为1其余为0, 每一维有两个特征
        data_arr.append([int(int(num) > 128) for num in cur_line[1:]])

    return data_arr, data_lable

def test(test_data_arr, test_lable, PYCk, PXxYCk):
    err_cnt = 0
    for i in range(len(test_data_arr)):
        x = test_data_arr[i]
        x_lable = test_lable[i]
        if naive_bayes_pred(PYCk, PXxYCk, x) != x_lable:
            err_cnt += 1

    return 1 - (err_cnt / len(data_lable))


def naive_bayes_pred(PYCk, PXxYCk, x):
    '''
    使用朴素贝叶斯进行分类
    :param PYCk: 先验概率
    :param PXxYCk: 条件概率
    :param U: 平滑值
    :param x: 需要进行预测的手写数字图片
    :return: 预测结果
    '''
    class_num = 10
    feature_num = 784
    P = [0] * class_num
    #全都已经转成log所以只需相加
    for i in range(class_num):
        sum = 0;
        for j in range(feature_num):
            sum += PXxYCk[i][j][x[j]]
        P[i] = sum + PYCk[i]

    return np.argmax(P[i])

def cnt_PYCk(data_lable, U):
    '''
    计算先验概率 P(Y=Ck)
    :param data_lable: 标签数组
    :param Ck 需要计算的事件
    :param U 为平滑参数
    :param class_num 分类数
    :return: 先验概率PYCk
    '''
    class_num = 10
    PYCk = np.zeros((class_num, 1))
    for i in range(class_num):
        PYCk[i] = np.log((np.sum(np.mat(data_lable) == i) + U) / (len(data_lable) + class_num * U))

    return PYCk

def cnt_PXxYCk(data_lable, data_arr, U):
    '''
    计算条件概率 P(X=x|Y=Ck) 在标签为Ck条件下向量为X的概率
    可转换为P(X1=x1, X2=x2,..., Xn=xn|Y=Ck)即向量Y=Ck条件下每一维为xn的概率相乘
    这边是假设事件独立
    :param data_lable:标签数组
    :param data_arr: 数据数组
    :param class_num 分类数
    :param U 平滑参数
    :return: PXxYCk 条件概率
    '''
    #维度数量
    feature_num = len(data_arr[0])
    #分类数
    class_num = 10

    #条件概率, 为三维矩阵, 其中2为两种特征0,1
    PXxYCk = np.zeros((class_num, feature_num, 2))
    #计算在lable条件下每个维度的各个特征总共有多少个
    for i in range(len(data_lable)):
        lable = data_lable[i]
        for j in range(feature_num):
            #特征为0或1
            x = data_arr[i][j]
            PXxYCk[lable][j][x] += 1

    #计算概率
    for i in range(class_num):
        for j in range(feature_num):
            PXxYCk0 = PXxYCk[i][j][0]
            PXxYCk1 = PXxYCk[i][j][1]
            #第i个lable的第j维的特征分别为0和1的概率
            #取对数防止下溢为0
            PXxYCk[i][j][0] = np.log((PXxYCk0 + U) / (PXxYCk0 + PXxYCk1 + U * 2))
            PXxYCk[i][j][1] = np.log((PXxYCk1 + U) / (PXxYCk0 + PXxYCk1 + U * 2))

    return PXxYCk

if __name__ == '__main__':
    data_arr, data_lable = load_data('../Mnist/mnist_train/mnist_train.csv')
    test_data_arr, test_lable = load_data('../Mnist/mnist_test/mnist_test.csv')
    start = time.time()
    print('开始训练')
    PYCk = cnt_PYCk(data_lable, 1)
    PXxYCk = cnt_PXxYCk(data_lable, data_arr, 1)
    print('开始测试')
    accuracy = test(test_data_arr, test_lable, PYCk, PXxYCk)
    print('准确率为：', accuracy)
    end = time.time()
    print('共花费：', end - start, 'ms')

