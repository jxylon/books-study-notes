## 函数功能

+ class LogisticRegression(object):逻辑斯谛回归类
  + def fit(self, x_, y_):训练
  + def predict(self, x_):预测
  + def gradient_descent(self, x_, y_, epsilon_=0.00001, n_iter=1500):梯度下降学习参数
+ def g(x_, y_, w_):经验风险函数
+ def sigmoid(x_):sigmoid
+ def load_data(path_='./train.csv'):加载数据