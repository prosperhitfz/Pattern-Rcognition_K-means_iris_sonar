import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
from sklearn.metrics import accuracy_score


k = 3       # k为聚类的类别数
total_numbers = 150     # n为样本总个数
feature_numbers = 4      # t为数据集的特征数

# FCM模糊聚类用到的参数
global MAX  # 用于初始化隶属度矩阵U
MAX = 10000
global Epsilon  # 结束阈值ε
Epsilon = 0.000001
a = 2  # 隶属度因子


def k_means():  # k-means算法
    means = np.zeros((k, feature_numbers))  # 初始化聚类中心
    means[0] = data[np.random.randint(0, total_numbers)]
    means[1] = data[np.random.randint(0, total_numbers)]
    means[2] = data[np.random.randint(0, total_numbers)]
    # 随机选取>=0且<total_numbers的三个值作为k（3）个初始聚类中心,聚类中心为每一类的均值向量

    # k_means聚类
    means_copy = means.copy()
    epoch = 0  # 初始化迭代次数
    while 1:
        means[0] = means_copy[0]
        means[1] = means_copy[1]
        means[2] = means_copy[2]  # 初始化每代的聚类中心
        w1 = np.zeros((1, feature_numbers))
        w2 = np.zeros((1, feature_numbers))
        w3 = np.zeros((1, feature_numbers))  # 初始化每一类的类别矩阵

        for i in range(total_numbers):
            distance = np.zeros(k)  # 初始化距离矩阵
            for j in range(k):
                distance[j] = np.linalg.norm(data[i] - means[j])  # 计算当前样本与每个聚类中心的欧式距离
            category = distance.argmin()  # 取距离最小值的下标为当前样本分类所属类别
            if category == 0:
                w1 = np.row_stack((w1, data[i]))
            if category == 1:
                w2 = np.row_stack((w2, data[i]))
            if category == 2:
                w3 = np.row_stack((w3, data[i]))
            # 将分类后的样本数据添加到对应类别矩阵中
        w1 = np.delete(w1, 0, axis=0)
        w2 = np.delete(w2, 0, axis=0)
        w3 = np.delete(w3, 0, axis=0)  # 删除原聚类中心
        means_copy[0] = np.mean(w1, axis=0)
        means_copy[1] = np.mean(w2, axis=0)
        means_copy[2] = np.mean(w3, axis=0)
        # 更新聚类中心

        # 若两代聚类中心相同，则聚类完成，迭代停止
        if (means[0] == means_copy[0]).all() and \
                (means[1] == means_copy[1]).all() and \
                (means[2] == means_copy[2]).all():
            print('kmeans第', epoch + 1, '次迭代：')
            print('父代和当前子代迭代设定的初始聚类中心均为：', means[0], means[1], means[2])
            print('迭代结束，kmeans聚类分类完毕\n')

            break
        else:
            print('kmeans第', epoch+1, '次迭代：')
            print('父代迭代设定的初始聚类中心为：', means[0], means[1], means[2])
            print('迭代后子代更新得到的聚类中心为：', means_copy[0], means_copy[1], means_copy[2], '\n')
        epoch += 1
        
        # 将每代得到的聚类结果给与标签并做标准（归一）化处理，为画图做准备
        w = np.vstack((w1, w2, w3))  # 将全部聚类结果归并为三行的矩阵
        # print(w)
        label1 = np.zeros((len(w1), 1))
        label2 = np.zeros((len(w2), 1))
        label3 = np.zeros((len(w3), 1))  # 初始化标签,第一类标签自动设置为0
        for i in range(len(w2)):
            label2[i, 0] = 1  # 将第二类标签设置为1
        for i in range(len(w3)):
            label3[i, 0] = 2  # 将第三类标签设置为2
        label = np.vstack((label1, label2, label3))
        label = np.ravel(label)  # 将矩阵000...111...222由三行变为一行（降维）
        y = label
        pca = decomposition.PCA(n_components=2)  # 四维数据降至二维
        normal_w = pca.fit_transform(w)  # 将得到的w数据拟合并标准化，方便后续数据可视化处理

        # 画出每一次迭代的聚类效果图
        fig = plt.figure()
        chart = fig.add_subplot(1, 1, 1)
        colors = ((1, 1, 0), (0, 1, 1), (1, 0, 1))
        for label, color in zip(np.unique(label), colors):
            position = y == label
            # print(position)
            chart.scatter(normal_w[position, 0], normal_w[position, 1], label=label, color=color)
        chart.set_xlabel("dimension1")
        chart.set_ylabel("dimension2")
        chart.set_title("K_means_iris_PCA")
        plt.show()
    return w1, w2, w3, y


def evaluate_result_purity_kmeans(label):  # 定义purity纯度为算法性能评估准则
    # 就是判断分类正确的个体占总个体数的百分比
    correct_label1 = np.zeros((50, 1))
    correct_label2 = np.zeros((50, 1))
    correct_label3 = np.zeros((50, 1))
    for i in range(len(correct_label2)):
        correct_label2[i, 0] = 1  # 将第二类标签设置为1
    for i in range(len(correct_label3)):
        correct_label3[i, 0] = 2  # 将第三类标签设置为2
    correct_label = np.vstack((correct_label1, correct_label2, correct_label3))
    correct_label = np.ravel(correct_label)
    # print(correct_label)
    # print(label)
    accuracy = accuracy_score(correct_label, label)  # 用到sklearn.metrics中的purity求解函数
    return accuracy * 100


def FCM_stop(U, U_copy):  # 设置算法停止条件
    global Epsilon  # 需要全局变量Epsilon作为设定的停止条件ε
    for i in range(len(U)):
        for j in range(len(U[0])):
            if abs(U[i][j] - U_copy[i][j]) > Epsilon:
                return False
    return True


def FCM():  # 模糊聚类算法
    global MAX  # 隶属度矩阵U的每行加起来都为1. 需要一个全局变量MAX
    global Epsilon  # 需要全局变量Epsilon作为设定的停止条件ε
    U = []
    for i in range(total_numbers):
        mid = []
        rand_sum = 0
        for j in range(k):
            fuzzy = np.random.randint(1, int(MAX))
            mid.append(fuzzy)
            rand_sum += fuzzy
        for j in range(k):
            mid[j] = mid[j] / rand_sum
        U.append(mid)      # 初始化隶属度矩阵U

    # 函数主体，计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    # 输入参数：簇数k、隶属度的因子a，其最佳取值范围为[1.5，2.5],这里取2

    while True:      # 循环更新隶属度矩阵U
        U_copy = np.copy(U)  # 创建它的副本，以检查结束条件
        fc_means = []
        for j in range(k):
            FC_Means = []  # 初始化聚类中心
            for i in range(len(data[0])):
                FC_Means_numerator = 0
                FC_Means_denominator = 0
                for loop in range(0, total_numbers):  # 聚类中心更新公式
                    # 分子
                    FC_Means_numerator += (U[loop][j] ** a) * data[loop][i]
                    # 分母
                    FC_Means_denominator += (U[loop][j] ** a)
                # 第i列的聚类中心
                FC_Means.append(FC_Means_numerator / FC_Means_denominator)
            # 第j类的所有聚类中心
            fc_means.append(FC_Means)

        # 创建一个距离向量, 用于更新隶属度矩阵U。
        distance_matrix = []
        for i in range(total_numbers):
            mid = []
            for j in range(k):
                res = 0
                for loop in range(len(data[i])):
                    # print(data[i][loop])
                    # print(fc_means[j][loop])
                    res += np.linalg.norm(data[i][loop] - fc_means[j][loop])
                mid.append(res)
            distance_matrix.append(mid)

        # 更新隶属度矩阵U
        for j in range(k):
            for i in range(0, total_numbers):
                U_denominator = 0
                for loop in range(k):  # 隶属度函数（矩阵）更新公式
                    # 分母
                    U_denominator += (distance_matrix[i][j] / distance_matrix[i][loop]) \
                                     ** (2 / (a - 1))
                U[i][j] = 1 / U_denominator

        # 标准化（模糊）隶属度矩阵U
        for i in range(len(U)):
            maximum = max(U[i])
            for j in range(len(U[0])):
                if U[i][j] != maximum:
                    U[i][j] = 0
                else:
                    U[i][j] = 1
        print('\nFCM聚类法经标准化后得到的隶属度矩阵U为：\n', U)

        # 数据可视化
        len1 = 0
        len2 = 0
        len3 = 0
        w1 = np.zeros((1, feature_numbers))
        w2 = np.zeros((1, feature_numbers))
        w3 = np.zeros((1, feature_numbers))  # 初始化每一类的类别矩阵
        for i in range(len(U)):
            if U[i] == [1, 0, 0]:
                len1 += 1
                w1 = np.row_stack((w1, data[i]))
            elif U[i] == [0, 1, 0]:
                len2 += 1
                w2 = np.row_stack((w2, data[i]))
            elif U[i] == [0, 0, 1]:
                len3 += 1
                w3 = np.row_stack((w3, data[i]))
        w1 = np.delete(w1, 0, axis=0)
        w2 = np.delete(w2, 0, axis=0)
        w3 = np.delete(w3, 0, axis=0)  # 删除原聚类中心
        w = np.vstack((w1, w2, w3))
        # 对每次迭代数据做可视化处理
        pca = decomposition.PCA(n_components=2)  # 四维数据降至二维
        normal_data = pca.fit_transform(w)  # 将数据拟合并标准化，方便后续数据可视化处理
        # print(normal_data)
        print('第一类数据个数：', len1)
        print('第二类数据个数：', len2)
        print('第三类数据个数：', len3)
        label1 = np.zeros((len1, 1))
        label2 = np.zeros((len2, 1))
        label3 = np.zeros((len3, 1))  # 初始化标签矩阵
        for i in range(len2):
            label2[i, 0] = 1
        for i in range(len3):
            label3[i, 0] = 2
        label = np.vstack((label1, label2, label3))
        label = np.ravel(label)
        y = label
        fig = plt.figure()
        chart = fig.add_subplot(1, 1, 1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for label, color in zip(np.unique(label), colors):
            position = y == label
            chart.scatter(normal_data[position, 0], normal_data[position, 1], label=label, color=color)
        chart.set_xlabel("dimension1")
        chart.set_ylabel("dimension2")
        chart.set_title("FCM_iris_PCA")
        plt.show()

        # 设定停止条件
        if FCM_stop(U, U_copy):
            print("已完成FCM模糊聚类")
            break
    return U, y


def evaluate_result_purity_FCM(label):  # 定义purity纯度为算法性能评估准则
    # 就是判断分类正确的个体占总个体数的百分比
    correct_label1 = np.zeros((50, 1))
    correct_label2 = np.zeros((50, 1))
    correct_label3 = np.zeros((50, 1))
    for i in range(len(correct_label2)):
        correct_label2[i, 0] = 1  # 将第二类标签设置为1
    for i in range(len(correct_label3)):
        correct_label3[i, 0] = 2  # 将第三类标签设置为2
    correct_label = np.vstack((correct_label1, correct_label2, correct_label3))
    correct_label = np.ravel(correct_label)
    # print(correct_label)
    # print(label)
    accuracy = accuracy_score(correct_label, label)  # 用到sklearn.metrics中的purity求解函数
    return accuracy * 100


# 数据预处理
iris = pd.read_csv('iris.data', header=None, sep=',')
iris1 = iris.iloc[0:150, 0:4]
data = np.mat(iris1)  # 转换为矩阵
# print(data)

if __name__ == '__main__':
    W1, W2, W3, LABEL_kmeans = k_means()
    print('第一类数据个数:', W1.shape[0])
    print('第二类数据个数:', W2.shape[0])
    print('第三类数据个数:', W3.shape[0])
    kmeans_acc = evaluate_result_purity_kmeans(LABEL_kmeans)
    print('kmeans聚类准确率:', kmeans_acc, '%')

    u, LABEL_FCM = FCM()
    FCM_Acc = evaluate_result_purity_FCM(LABEL_FCM)
    print('FCM聚类准确率：', FCM_Acc, '%')
