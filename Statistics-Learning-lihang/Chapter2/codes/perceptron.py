import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

# 点集
ori_data = pd.DataFrame(data=[[3, 3, 1], [4, 3, 1], [1, 1, -1]], columns=['x1', 'x2', 'y'])
# 维度
dimension = ori_data.shape[1] - 1
# 点的个数
point_len = ori_data.shape[0]
# 记录直线
x_list = []
y_list = []


def draw_formula(ori_data, w, b):
    # 点集
    points = ori_data.iloc[:, 0:dimension]
    # 图例和图集
    label_list = []
    plot_list = []
    # 画出点
    plt.plot(points['x1'], points['x2'], 'ro')
    # 画出直线
    x = np.linspace(1, 10)
    # x*w0 + y*w1 + b = 0
    if (w[1] == 0):
        if (w[0] != 0):
            y = np.linspace(-10, 10)
            x = [-1 * b / w[0]] * len(y)
        else:
            x = [0]
            y = [0]
    else:
        y = -1 * (x * w[0] + b) / w[1]
    x_list.append(x)
    y_list.append(y)
    for i in range(len(y_list)):
        p, = plt.plot(x_list[i], y_list[i])
        plot_list.append(p)
        label = 'line' + str(i + 1)
        label_list.append(label)
    plt.legend(handles=plot_list, labels=label_list, loc='upper right')
    plt.show()


def algorithm(ori_data, rate):
    """
    :param ori_data:点集
    :param rate: 学习率
    :return: 超平面
    """
    # 初始化误分类标记位
    ori_data['is_miss'] = 1
    # 初始化参数w，b
    w = np.zeros((dimension, 1))  # 列向量
    b = 0
    # 记录的下标
    index = 0
    # 直至没有误分点
    while (len(ori_data[ori_data['is_miss'] == 0]) != point_len):
        # 输出参数信息
        print('round %d' % (index), end=',')
        for i in range(dimension):
            print('w(%d) = %d' % (i, w[i][0]), end=',')
        print('b = %d' % (b), end=',')
        # 随机种子点
        rdn_seed = random.randint(0, point_len - 1)
        # 确保找到误分点
        while (ori_data.loc[rdn_seed, 'is_miss'] == 0):
            rdn_seed = random.randint(0, point_len - 1)
        # x,y,L(w,b)
        xi_vec = np.array(ori_data.iloc[rdn_seed, 0:dimension]).reshape(1, -1).T  # 列向量
        yi = ori_data.loc[rdn_seed, 'y']
        print('选择点x%d(%d,%d),y=%d' % (rdn_seed, xi_vec[0], xi_vec[1], yi))
        formula = yi * (xi_vec.T.dot(w)[0][0] + b)
        # 若formula>0,不是误分点
        if (formula > 0):
            ori_data.loc[rdn_seed, 'is_miss'] = 0
            print('x%i-非误分点' % (rdn_seed))
            continue
        else:
            # 随机梯度下降
            w = w + rate * yi * xi_vec
            b = b + rate * yi
            ori_data['is_miss'] = 1
            index += 1
            print('x%i-误分点' % (rdn_seed))
            draw_formula(ori_data, w, b)


if __name__ == '__main__':
    algorithm(ori_data, 1)
