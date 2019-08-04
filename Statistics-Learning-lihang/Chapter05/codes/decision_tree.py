import pandas as pd
import numpy as np
import math
import treelib


# 书上题目5.1
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    datasets = np.array(datasets)
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    data_df = pd.DataFrame(datasets, columns=labels)
    # 返回数据集和每个维度的名称
    return data_df


# 定义节点类
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root  # 是否是叶子节点
        self.label = label  # 标签
        self.feature_name = feature_name  # 特征名
        self.feature = feature  # 特征值
        self.tree = {}  # 子树

    def __repr__(self):
        result = {
            'label:': self.label,
            'feature': self.feature,
            'feature_name': self.feature_name,
            'tree': self.tree
        }
        return '{}'.format(result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class Decision_Tree(object):
    def __init__(self, data_df, epsilon):
        self.columns = list(data_df.columns)
        self.epsilon = epsilon
        self.tree = {}

    # 计算经验熵
    def cal_exp_entropy(self, data_df):
        D_len = len(data_df)
        y_set = set(data_df.loc[:, '类别'])
        res = 0
        for y_elm in y_set:
            Ck = len(data_df[data_df['类别'] == y_elm])
            tmp_res = -1 * Ck / D_len * math.log2(Ck / D_len)
            res += tmp_res
        return res

    # 计算给定数据集下，给定特征的信息增益
    def cal_info_gain(self, data_df):
        info_gain_list = []
        for feature in data_df.columns[:-1]:
            # 经验熵
            exp_entropy = self.cal_exp_entropy(data_df)
            # 经验条件熵
            D_len = len(data_df)
            feature_set = set(data_df[feature])
            y_set = set(data_df.loc[:, '类别'])
            exp_con_entropy = 0
            for fea_elm in feature_set:
                data_fea_df = data_df[data_df[feature] == fea_elm]
                Di_len = len(data_fea_df)
                res_fea = 0
                for y_elm in y_set:
                    data_y_fea_df = data_fea_df[data_fea_df['类别'] == y_elm]
                    Dik_len = len(data_y_fea_df)
                    if (Dik_len == 0):
                        log_tmp = 0
                    else:
                        log_tmp = math.log2(Dik_len / Di_len)
                    res_fea += -1 * (Dik_len / Di_len) * log_tmp
                res_fea = res_fea * Di_len / D_len
                exp_con_entropy += res_fea
            # 信息增益
            info_gain = exp_entropy - exp_con_entropy
            info_gain_list.append(info_gain)
        max_info_gain = max(info_gain_list)
        max_feature_name = data_df.columns[np.argmax(info_gain_list)]
        max_feature = self.columns.index(max_feature_name)
        return max_feature, max_feature_name, max_info_gain

    # 构造ID3决策树
    def train(self, data_df):
        # 1.如果数据集中的类均是同一类，则归为叶结点
        if (len(data_df.iloc[:, -1].unique()) == 1):
            return Node(root=True, label=data_df.iloc[:, -1].unique()[0])
        # 2.如果没有特征可分，归为叶结点，类别为出现次数最多的类别
        if (len(data_df.columns) == 1):
            return Node(root=True, label=data_df.iloc[:, -1].value_counts().sort_values(
                ascending=False).index[0])
        # 计算信息增益特征与信息增益值
        max_feature, max_feature_name, max_info_gain = self.cal_info_gain(data_df)
        # 3.如果信息增益值小于阈值，归为叶结点，类别为出现次数最多的类别
        if max_info_gain < self.epsilon:
            return Node(root=True, label=data_df.iloc[:, -1].value_counts().sort_values(
                ascending=False).index[0])
        # 4.构建子树
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)
        feature_list = data_df.loc[:, max_feature_name].unique()
        # 循环max_feature取值
        for fea in feature_list:
            sub_data_df = data_df[data_df[max_feature_name] == fea].drop([max_feature_name], axis=1)
            # 5.递归生成树
            sub_tree = self.train(sub_data_df)
            node_tree.add_node(fea, sub_tree)
        self.tree = node_tree
        return node_tree

    def predict(self, X_test):
        fea_index = self.tree.feature
        fea_value = X_test[fea_index]
        next_tree = self.tree.tree[fea_value]
        while (not next_tree.root):
            fea_index = next_tree.feature
            fea_value = X_test[fea_index]
            next_tree = next_tree.tree[fea_value]
        return next_tree.label


def stack_pop(parent_stack):
    x = parent_stack[-1][0]
    time = parent_stack[-1][1]
    if (time == 1):
        parent_stack.pop(-1)
    else:
        parent_stack[-1][1] = time - 1
    return x, parent_stack


def draw_tree(dt_node: Node):
    tl = treelib.Tree()
    node_queue = [dt_node]
    parent_stack = [(None, 1)]
    node_id = 0
    while (node_queue):
        tree = node_queue.pop(0)
        if (not tl.contains('nid' + str(tree.feature))):
            parent, parent_stack = stack_pop(parent_stack)
            tl.create_node(tree.feature_name, 'nid' + str(tree.feature), parent=parent)
            parent_stack.append(['nid' + str(tree.feature), 2])
        tmp_tree = tree.tree
        for k in tmp_tree.keys():
            next_tree = tmp_tree[k]
            if (next_tree.label != None):
                parent, parent_stack = stack_pop(parent_stack)
                tl.create_node('(' + k + ')' + str(next_tree.label), 'lid' + str(node_id), parent=parent)
                node_id += 1
            else:
                parent, parent_stack = stack_pop(parent_stack)
                tl.create_node('(' + k + ')' + str(next_tree.feature_name), 'nid' + str(next_tree.feature),
                               parent=parent)
                parent_stack.append(['nid' + str(next_tree.feature), 2])
                node_queue.append(next_tree)
    tl.show()


if __name__ == '__main__':
    data_df = create_data()
    dt = Decision_Tree(data_df, 0.1)
    dt_node = dt.train(data_df)
    draw_tree(dt_node)
    y_test = dt.predict(['青年', '是', '是', '一般'])
    print(y_test)