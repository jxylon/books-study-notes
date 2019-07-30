import numpy as np
from collections import Counter
import math


class bayes(object):
    def __init__(self):
        self.prior_dic = None
        self.condic_dic = None

    def fit(self, data):
        print("Training ...")
        data = np.array(data, dtype=object)  # 转numpy
        N = data.shape[0]  # xi数量
        dim = data.shape[1] - 1  # 维度
        data_x = data[:, 0:-1]  # 训练数据x
        data_y = data[:, -1]  # 类别
        # 1 计算各类的先验概率
        prior_dic = {}  # 先验概率
        for y_elm in set(data_y):
            prior_dic[y_elm] = len(data_y[data_y == y_elm]) / N  # 各类标签个数 / 总个数
        # 2 计算各类的条件概率
        # 2.1 求维度上的各元素个数
        condic_dic = {}  # 条件概率
        for y_elm in set(data_y):
            dim_dic = {}  # 不同维度下的存储字典
            tmp_data_x = data_x[data_y == y_elm]
            for dim_elm in range(dim):
                xi_dic = {}  # 该维度下不同元素存储字典
                tmp_dim_x = tmp_data_x[:, dim_elm]
                for xi_elm in set(tmp_dim_x):
                    xi_dic[xi_elm] = len(tmp_dim_x[tmp_dim_x == xi_elm]) / len(tmp_dim_x)  # 该维度下各类元素个数 / 该维度下总个数
                dim_dic[dim_elm] = xi_dic
            condic_dic[y_elm] = dim_dic
        self.prior_dic = prior_dic
        self.condic_dic = condic_dic
        print("Training has done...")

    def predict(self, test_data):
        test_data = np.array(test_data, dtype=object)  # 转numpy
        N = test_data.shape[0]  # test_x数量
        dim = test_data.shape[1]  # 维度
        test_y = []
        for n in range(N):
            test_xi = test_data[n]
            tmp_prob = []
            # 各类先验概率 * 不同维度下的条件概率
            for y in self.prior_dic.keys():
                tmp_y_prob = 1  # 各类的概率
                for d in range(dim):
                    tmp_y_prob *= self.condic_dic[y][d][test_xi[d]]
                tmp_prob.append(tmp_y_prob)
            # 该向量属于后验概率最大的类
            test_y.append(list(self.prior_dic.keys())[int(np.argmax(tmp_prob))])
        return np.array(test_y, dtype=object)


if __name__ == '__main__':
    # 书上例4.1数据
    train_data = np.array(
        [[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1], [2, 'S', -1], [2, 'M', -1],
         [2, 'M', 1], [2, 'L', 1], [2, 'L', 1], [3, 'L', 1], [3, 'M', 1], [3, 'M', 1],
         [3, 'L', 1], [3, 'L', -1]], dtype=object)
    obj_bayes = bayes()
    obj_bayes.fit(train_data)
    print("训练的先验概率分布")
    print(obj_bayes.prior_dic)
    print("训练的条件概率分布")
    print(obj_bayes.condic_dic)
    test_x = [[2, 'S'], [1, 'M']]
    print("预测数据为：")
    print(test_x)
    test_y = obj_bayes.predict(test_x)
    print("预测结果为：")
    print(test_y)
