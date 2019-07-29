from math import sqrt
from collections import namedtuple
import treelib

result = namedtuple("Reulst_tuple", "nearest_point  nearest_dist  nodes_visited")


# kd树结点类
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right


# kd树类
class KdTree(object):
    def __init__(self, data):
        # k维度
        self.k = len(data[0])
        self.data = data

    def CreateKdTree(self, split, data_set):
        if (not data_set):
            return None
        # 对split维排序
        data_set.sort(key=lambda x: x[split])
        # 求出中位数
        median_pos = len(data_set) // 2
        median = data_set[median_pos]
        # 下一个要排序的维度
        split_next = (split + 1) % self.k
        # 递归创建kd树
        return KdNode(median,
                      split,
                      self.CreateKdTree(split_next, data_set[0:median_pos]),
                      self.CreateKdTree(split_next, data_set[median_pos + 1:]
                                        ))


def CreateDrawTree(kdtree: KdNode):
    # treelib对象
    tree = treelib.Tree()
    # 队列
    queue = []
    queue.append([kdtree, None])
    # 父节点
    # 层次遍历
    while (queue):
        node, parent = queue.pop(0)
        print(node.dom_elt, parent)
        tree.create_node(str(node.dom_elt), str(node.dom_elt), parent=parent)
        if (node.left):
            queue.append([node.left, str(node.dom_elt)])
        if (node.right):
            queue.append([node.right, str(node.dom_elt)])
    return tree


def searchKdTree(tree, point):
    """
    最近邻法步骤
    1） 找到目标点的根结点，将其作为最近点
    2） 回溯到父结点
        a) 如果父结点到目标点的距离小于最近点到目标点距离，更新最近点为父结点
        b) 以目标点为圆心，目标点到最近点距离为半径，画圆
            如果与父结点另一个区域相交，移动到另一个区域
    3）重复到2），直至没有结点可以回溯
    :return:
    """
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"),
                          0)  # python中用float("inf")和float("-inf")表示正负无穷
        # 结点访问次数
        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left
        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断
        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree, point, float('inf'))


if __name__ == '__main__':
    # 原始点
    obj_KdTree = KdTree([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    # 构造kd树
    kdtree = obj_KdTree.CreateKdTree(0, obj_KdTree.data)
    # 创建treelib树
    drawTree = CreateDrawTree(kdtree)
    drawTree.show()
    ret = searchKdTree(kdtree, [3, 4.5])
    print(ret)
