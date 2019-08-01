## 函数功能

+ def create_data()：创建数据集

+ class Node:节点类

+ class Decision_Tree(object):决策树类

  + def cal_exp_entropy(self, data_df): 计算经验熵
  + def cal_info_gain(self, data_df): 计算给定数据集下，给定特征的信息增益
  + def train(self, data_df): 构造ID3决策树
  + def predict(self, X_test): 预测

+ def draw_tree(dt_node: Node): 画出决策树结构，如下所示

  > 有自己的房子
  > ├── (否)有工作
  > │   ├── (否)否
  > │   └── (是)是
  > └── (是)是