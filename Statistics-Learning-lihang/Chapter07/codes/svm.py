import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random


class SVM(object):
    def __init__(self, X, y, tol=10e-3, C=0.6, n_iters=10, verbose=True):
        self.X = X
        self.y = y
        self.tol = tol
        self.C = C
        self.verbose = verbose
        self.m = len(X)
        self.n_iters = n_iters
        self.alpha = np.zeros(self.m)
        self.b = 0

    def fit(self):
        self.smo()
        w_ = np.dot(X.T,self.alpha*self.y)
        tmp = pd.Series(self.alpha)
        j = tmp.where(tmp > 0).index[0]
        b_ = self.y[j] - np.sum(self.alpha*self.y*np.dot(self.X,self.X[j,:]),axis=0)
        self.w_ = w_
        self.b_ = b_

    def predict(self,X):
        res = np.dot(X,self.w_) + self.b_
        return 1 if res < 0 else -1

    def smo(self):
        # 迭代n_iters次
        n_iter = 0
        while (n_iter < self.n_iters):
            alpha_pairs_changed = 0
            for i in range(self.m):
                if(self.verbose):
                    print("iter:%d,i:%d" % (n_iter, i), end=",")
                # 计算e_i
                e_i = self.cal_e_i(i)
                # 限制条件
                if (((self.y[i] * e_i < -self.tol) and (self.alpha[i] < self.C)) or (
                        (self.y[i] * e_i > self.tol) and (self.alpha[i] > 0))):
                    # 随机选取一个j
                    j = self.select_j(i)
                    # 计算e_j
                    e_j = self.cal_e_i(j)
                    # 保存alpha_old
                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()
                    # 得到上下界
                    if (self.y[i] != self.y[j]):
                        L = max(0.0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C+alpha_j_old - alpha_i_old)
                    else:
                        L = max(0.0, alpha_j_old + alpha_i_old - self.C)
                        H = min(self.C, alpha_j_old + alpha_i_old)
                    if (L == H):
                        if (self.verbose):
                            print("L==H")
                        continue
                    # 计算eta
                    eta = self.cal_eta(i, j)
                    if (eta <= 0):
                        if (self.verbose):
                            print("eta<=0,%f" % eta)
                        continue
                    # 计算alpha_new_unc
                    self.alpha[j] += (self.y[j] * (e_i - e_j)) / eta
                    # 增加不明显
                    if (abs(self.alpha[j] - alpha_j_old) < 0.00001):
                        if (self.verbose):
                            print("j not moving enough!")
                        continue
                    # 剪辑alpha_j
                    self.alpha[j] = self.cut(j, L, H)
                    # 根据alpha_j 得到 alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    # 计算b_new
                    b_i_new = self.b - e_i - self.y[i] * np.dot(self.X[i, :], self.X[i, :]) * (self.alpha[i] - alpha_i_old) - \
                              self.y[j] * np.dot(self.X[j, :], self.X[i, :]) * (self.alpha[j] - alpha_j_old)
                    b_j_new = self.b - e_j - self.y[i] * np.dot(self.X[i, :], self.X[j, :]) * (self.alpha[i] - alpha_i_old) - \
                              self.y[j] * np.dot(self.X[j, :], self.X[j, :]) * (self.alpha[j] - alpha_j_old)
                    # 更新b
                    if (0 < self.alpha[i]) and (self.C > self.alpha[j]):
                        self.b = b_i_new
                    elif (0 < self.alpha[j]) and (self.C > self.alpha[j]):
                        self.b = b_j_new
                    else:
                        self.b = (b_i_new + b_j_new) / 2
                    alpha_pairs_changed += 1
                    if (self.verbose):
                        print("pairs_changed:%d" % alpha_pairs_changed, end='')
                if(self.verbose):
                    print()
            if (alpha_pairs_changed == 0):
                n_iter += 1
            else:
                n_iter = 0
            if (self.verbose):
                print("iteration:%d" % n_iter)

    def cal_g_i(self, i):
        g_i = np.sum(self.alpha * self.y * np.dot(self.X, self.X[i, :]),axis=0) + self.b
        return g_i

    def cal_e_i(self, i):
        e_i = self.cal_g_i(i) - self.y[i]
        return e_i

    def select_j(self, i):
        j = i
        while (j == i):
            j = random.randint(0, self.m - 1)
        return j

    def cal_eta(self, i, j):
        return np.dot(self.X[i, :], self.X[i, :]) + np.dot(self.X[j, :], self.X[j, :]) - 2 * np.dot(self.X[i, :], self.X[j, :])

    def cut(self, j, L, H):
        if (self.alpha[j] > H):
            return H
        elif (self.alpha[j] < L):
            return L
        else:
            return self.alpha[j]


def create_data():
    iris_data = load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['label'] = iris_data.target
    # 100行，3个特征加上1个标签
    data = np.array(iris_df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # 返回 X,y
    return data[:, :2], data[:, -1]


def draw_data(X, y):
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='1')
    plt.legend()
    plt.show()

def draw_line(X,y,w,b):
    x_axis = np.linspace(4,7)
    y_axis = -1 * (x_axis * w[0] + b) / w[1]
    plt.plot(x_axis,y_axis)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='1')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 创建数据
    X, y = create_data()
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # 画出散点图
    draw_data(X, y)
    # 创建对象
    svm = SVM(X,y,verbose=False)
    # 训练
    svm.fit()
    # 画线
    draw_line(X,y,svm.w_,svm.b_)
    # 预测
    y_pred = svm.predict([1,1])