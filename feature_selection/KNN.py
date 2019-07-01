# from sklearn import datasets  # 导入数据
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import urllib
import ssl


class KnnTest:
    """一个作为评价特征选择FSS性能的Wrapper(KNN分类器)
    :param url: 使用的训练数据的网址
    :param option: 1从本地加载数据，0从远程连接加载数据
    """
    def __init__(self, url, option=0):
        if option == 0:
            data_set = self.data_online_loader(url)
        else:
            data_set = self.data_local_loader(url)
        self.feature_size = data_set[0].size-1

        # 从数据中分出用于训练特征选择器时的knn训练与测试数据和测试特征选择器性能时knn的训练与测试数据
        train_data, test_data = train_test_split(data_set, test_size=0.3)
        self.train_train, self.train_test = train_test_split(train_data, test_size=0.3)
        self.test_train, self.test_test = train_test_split(test_data, test_size=0.3)

    def data_online_loader(self, url):
        # 生成证书上下文,即不验证https证书请求数据
        context = ssl._create_unverified_context()
        raw_data = urllib.request.urlopen(url, context=context)
        dataset = np.loadtxt(raw_data, delimiter=",")
        return dataset

    def data_local_loader(self, url):
        f1 = open(url)
        dataset = f1.readlines()
        dataset = self.data_pre_processing(dataset)
        dataset = np.array(dataset)
        # wdbc special
        # dataset = dataset[:, 1:]
        return dataset

    def data_pre_processing(self, dataset):
        for i in range(len(dataset)):
            # wdbc special
            # dataset[i] = dataset[i].replace('M', '0')
            # dataset[i] = dataset[i].replace('B', '1')

            # wine special
            # dataset[i] = dataset[i][0:-2]

            # iris special
            dataset[i] = dataset[i].replace('Iris-setosa', '0')
            dataset[i] = dataset[i].replace('Iris-versicolor', '1')
            dataset[i] = dataset[i].replace('Iris-virginica', '2')

            dataset[i] = dataset[i].split(',')
            for j in range(len(dataset[i])):
                dataset[i][j] = float(dataset[i][j])

            # iris special
            temp = dataset[i][0]
            dataset[i][0] = dataset[i][-1]
            dataset[i][-1] = temp

        return dataset

    def feature_sel(self, individual, data_x):
        num = 0
        for fea in individual:
            if fea > 0.6:
                num = num + 1
            else:
                data_x = np.delete(data_x, num, axis=1)
        return num, data_x

    def knnTesting(self, individual, n_neighbors=5, option=0):
        """
        knn_traindatax,knn_traindatay: 用于训练knn的训练集
        knn_testdatax,knn_testdatay: 用于测试knn的测试集
        option == 0: 使用训练特征选择器的数据
        option == 1: 使用测试最终选择器效果的数据
        """
        if option == 0:
            knn_traindata = self.train_train
            knn_testdata = self.train_test
        else:
            knn_traindata = self.test_train
            knn_testdata = self.test_test

        # 数据大小
        dataSize = knn_traindata[0].size

        # 把数据根据knn的训练与测试需要再分为训练集与测试集
        knn_traindatax, knn_traindatay = knn_traindata[:, 1:dataSize], knn_traindata[:, 0]
        knn_testdatax, knn_testdatay = knn_testdata[:, 1:dataSize], knn_testdata[:, 0]

        # 删除没有选中的特征
        feature_number, knn_traindatax = self.feature_sel(individual, knn_traindatax)
        feature_number, knn_testdatax = self.feature_sel(individual, knn_testdatax)

        # 排除特殊情况
        if feature_number == 0:
            return 0

        # knn模型调用
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(knn_traindatax, knn_traindatay)
        y_pre = knn.predict(knn_testdatax)

        # 返回成功率, 1-score后得到错误率
        score = metrics.accuracy_score(knn_testdatay, y_pre)
        # error_rate = 1 - score

        return score
