from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


class LogisticRegression(object):

    def __init__(self,
                 learning_step=0.0001,
                 epsilon=0.001,
                 n_iter=1500):
        self.learning_step = learning_step
        self.epsilon_ = epsilon
        self.n_iter_ = n_iter
        self.coef_ = np.array([])
        self.cols_ = []

    def fit(self, x_, y_):
        return self.gradient_descent(x_, y_, epsilon_=self.epsilon_, n_iter=self.n_iter_)

    def predict(self, x_):
        rst = np.array([self.cols_[idx] for idx in [np.argmax(rst) for rst in sigmoid(np.dot(x_, self.coef_.T))]])
        return rst

    def gradient_descent(self, x_, y_, epsilon_=0.00001, n_iter=1500):
        n = x_.shape[len(x_.shape) - 1]
        # one-hot encoding
        y_ = pd.get_dummies(y_)
        w_ = np.array([])
        print(n, y_.shape, y_.columns)

        # 多分类模型转为多个二分类模型
        for ck in np.arange(y_.shape[1]): # y.shape[1]种类别
            wck_ = np.zeros(n) # 某一类的w参数
            for k in np.arange(n_iter): # 迭代n_iter次
                g_k = self.g(x_, y_.values[:, ck], wck_) # 损失函数
                if np.average(g_k * g_k) < epsilon_: # 若损失小于阈值，则提前结束迭代
                    w_ = wck_ if w_.size == 0 else np.vstack([w_, wck_]) # 加入结果
                    break
                else:
                    p_k = - g_k
                lambda_k = 0.0000001  # TODO: 更新算法
                wck_ = wck_ + lambda_k * p_k # 梯度下降
            if k == n_iter - 1: # 迭代n_iter次还没收敛，加入结果
                w_ = wck_ if w_.size == 0 else np.vstack([w_, wck_])
            print("progress: %d done" % ck)
        self.coef_ = w_
        self.cols_ = y_.columns.tolist()
        return self.coef_, self.cols_


def g(x_, y_, w_):
    m = y_.size
    rst_ = -(1 / m) * np.dot(x_.T, y_ - sigmoid(np.dot(x_, w_)))
    return rst_


def sigmoid(x_):
    p = np.exp(x_)
    p = p / (1 + p)
    return p


def load_data(path_='./train.csv'):
    """
    data size is 28x28, 784
    :param path_:
    :return:
    """
    raw_data = pd.read_csv(path_)
    y = raw_data["label"].values
    del raw_data["label"]
    X = raw_data.values
    return X, y


if __name__ == "__main__":
    print('Start read data')
    X, y = load_data()
    X = X[:300]
    y = y[:300]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=2018)

    print('Start training')
    clf = LogisticRegression()
    clf.g = g
    clf.fit(train_x, train_y)

    print('Start predicting')
    test_predict = clf.predict(test_x)

    score = accuracy_score(test_y, test_predict)
    print("The accruacy socre is ", score)
