## 函数功能

+ class SVM(object):支持向量机类
  + def fit(self, x_, y_):训练
  + def smo(self, x_):smo算法
  + def predict(self, X):预测
  + def cal_g_i(self, i):计算g_i
  + def cal_e_i(self, i):计算e_i
  + def select_j(self, i):随机选择不等于i的j
  + def cal_eta(self, i, j):计算eta
  + def cut(self, j, L, H):对α2剪裁
+ def create_data():创建数据
+ def draw_data(X, y):画散点图
+ def draw_line(X,y,w,b):画训练后的图