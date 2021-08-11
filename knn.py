import numpy as np
import pandas as pd
import time

def load_data(filePath):
    print('开始读取数据')
    df = pd.read_csv(filePath)
    #第一个冒号代表索引，第二个代表每个向量对应的数据分片
    data_arr = df.iloc[:, 1:]
    data_lable = df.iloc[:, 0]

    return data_arr, data_lable

def cal_dist(x, x1):
    '''
    计算两点的欧氏距离
    :param x: 第一个点
    :param x1: 第二个点
    :return: 两点间的欧氏距离
    '''
    return np.sqrt(np.sum(np.square(x - x1)))

def get_topK_lable(train_data, train_lable, x, topK):
    '''
    获取x在样本中最近的K个点分别对应相同标签的个数
    :param train_data: 样本数据
    :param train_lable: 样本标签
    :param x: 测试数据点
    :param topK: 最邻近的k个
    :return: 样本里测试点最近的K个点的标签对应个数
    '''
    train_data_mat = np.mat(train_data)
    train_lable_mat = np.mat(train_lable).T
    #x和样本中所有点的距离，扩充为长度为所有样本集的长度
    dist = [0] * len(train_lable_mat)
    #计算x和样本集中所有点的距离
    for i in range(len(train_data_mat)):
        xi = train_data_mat[i]
        xdist = cal_dist(x, xi)
        dist[i] = xdist
    #top_k_lable[0]个数为邻近0标签点的个数
    top_k_lable = [0] * topK
    #np.argsort返回排序后的索引位置，索引位置为该点在样本集中的位置
    top_k_index = np.argsort(np.array(dist))[:topK]
    #计算topk点对应的lable的计数
    for i in top_k_index:
        top_k_lable[int(train_lable_mat[i])] += 1

    return top_k_lable

def test(train_data, train_lable, test_data, test_lable, topK=25, test_size=100):
    #将测试集转换为矩阵以便使用
    print('开始测试')
    test_data_mat = np.mat(test_data)
    test_lable_mat = np.mat(test_lable).T
    errCnt = 0
    for i in range(test_size):
        print('共%d轮，正进行第%d轮' % (test_size, i))
        x = test_data_mat[i]
        top_k_lable = get_topK_lable(train_data, train_lable, x, topK)
        final_lable = np.argsort(np.array(top_k_lable))[-1]
        if final_lable != test_lable_mat[i]:
            errCnt += 1
    return 1 - (errCnt / test_size)

if __name__ == '__main__':
    train_data, train_lable = load_data('./Mnist/mnist_train/mnist_train.csv')
    test_data, test_lable = load_data('./Mnist/mnist_test/mnist_test.csv')
    start = time.time()
    print('准确率为：%f' % test(train_data, train_lable, test_data, test_lable))
    end = time.time()
    print('共花费时间为%ds' % (end - start))