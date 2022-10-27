import numpy as np
import random as rn

class iForest(object):
    """
    创建iForest对象
    该对象保存数据以及经过训练的树（iTree对象）

    属性
    X(list): 用于训练的数据。这是一个浮动列表。
    nobjs(int)：数据集的大小。
    sample(int)：用于创建树的样本的大小。
    tree(list)：树对象的列表。
    limit(int)：树的最大深度。
    exlevel(int)：要在创建拆分标准中使用的注释级别。
    c(float)：用于计算异常分数的乘法因子。
    """

    def __init__(self, X, ntrees, sample_size, limit=None, ExtensionLevel=0):
        """
        通过传递训练数据、要使用的树数和子样本大小来初始化林

        参数
        X(list of list of floats):训练数据,坐标点列表[x1,x2,...,xn]
        ntrees(int):要使用的树的数量
        sample_size(int):用于创建每棵树的子样本的大小,必须小于X
        limit(int):允许的最大树深度.默认情况下，该值设置为二叉树中未成功搜索的平均长度。
        ExtensionLevel(int):指定选择用于分割数据的超平面的自由度.必须小于数据集的维度n
        """

        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.exlevel = ExtensionLevel
        # 检查扩展级别不超过数据的维度
        dim =self.X.shape[1]
        if self.exlevel < 0 or self.exlevel > dim - 1:
            raise Exception("ExtensionLevel is not canonical")
        # 将limit设置指定的默认值
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))
        # c为一个包含n个样本的数据集，树的平均路径长度，用来标准化记录x的路径长度。H(*)为调和数，ξ为欧拉常数，约为0.5772156649
        # 用来计算异常分数
        self.c = 2.0 * (np.log(self.sample - 1) + 0.5772156649) - (2.0 * (self.sample - 1.) / (self.sample * 1.0))
        # 构建iTrees（森林）集合
        for i in range(self.ntrees):
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(iTree(X_p, 0, self.limit, exlevel=self.exlevel))

    def compute_paths(self, X_in=None):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i], self.Trees[j]).path * 1.0  # 计算每个点的路径长度
            Eh = h_temp / self.ntrees  # 所有树中该点的平均路径长度。
            S[i] = 2.0 ** (-Eh / self.c)  # 异常分数
        return S


class Node(object):
    """
    每个树（每个iTree对象）的单个节点。节点包含用于数据分割的超平面上的信息，以及要传递给左右节点的数据，是外部节点还是内部节点。

    属性
    e(int):节点所属树的深度
    size(int):节点上存在的数据集的大小
    X(list):节点上的数据集
    n(list):用于构建分割节点中数据的超平面的法向量
    p(list):超平面通过的截距
    left(Node object):左子节点
    right(Node object):右子节点
    ntype(str):节点的类型为“exNode”或“inNode”。
    """

    def __init__(self, X, n, p, e, left, right, node_type=''):
        self.e = e
        self.size = len(X)
        self.X = X
        self.n = n
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type


class iTree(object):
    """
    森林中使用唯一子样本构建的一棵树.

    属性
    exlevel(int):拆分条件中使用的扩展级别
    e(int):树的深度
    X(list):此树的根节点上存在的数据集
    size(int):数据集大小
    dim(int):数据集的维度
    l(int):树在创建结束前可以达到的最大深度
    n(list):此树根的法向量，用于创建用于拆分准则的超平面
    p(list):在这棵树的根上截距点，分裂的超平面通过它
    exnodes(int):此树具有的叶子节点数
    root(Node object):在每个节点上创建一棵新树.

    方法
    make_tree(X, e, l)：从给定节点递归生成树。返回一个节点对象.
    """

    def __init__(self, X, e, l, exlevel=0):
        # 初始化一棵树
        self.exlevel = exlevel
        self.e = e
        self.X = X
        self.size = len(X)
        self.dim = self.X.shape[1]
        self.l = l
        self.p = None  # 截取用于在给定节点上拆分数据的超平面
        self.n = None  # 用于在给定节点拆分数据的超平面的法向量
        self.exnodes = 0
        self.root = self.make_tree(X, e, l)  # 在每个节点上创建一棵新树，从根节点开始

    def make_tree(self, X, e, l):
        # 从给定节点递归生成树。返回一个节点对象
        self.e = e
        if e >= l or len(X) <= 1:  # 在传输数据中隔离了一个点，或者达到了深度限制
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.n, self.p, e, left, right, node_type='exNode')
        else:  # 继续建造这棵树.所有这些节点都是内部节点
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            self.p = np.random.uniform(mins, maxs)  # P为超平面分割数据选择一个随机截取点。
            idxs = np.random.choice(range(self.dim), self.dim - self.exlevel - 1,replace=False)  # 根据扩展级别选择法向量元素应设置为零的索引
            self.n = np.random.normal(0, 1, self.dim)  # 从一个均匀的n-球体中选取的随机法向量。为了从n-球体均匀拾取，需要为该向量的每个分量拾取一个随机法线。
            self.n[idxs] = 0
            w = (X - self.p).dot(self.n) < 0  # 用于确定数据点应转到左侧还是右侧子节点的条件
            return Node(X, self.n, self.p, e, left=self.make_tree(X[w], e + 1, l), right=self.make_tree(X[~w], e + 1, l), node_type='inNode')


class PathFactor(object):
    """
    给定一棵树（iTree objext）和一个数据点x=[x1，x2，…，xn]，计算给定点到达叶子节点时所经过路径的长度。

    属性
    x(list):单个数据点，表示为浮点数列表。
    e(int):树中给定节点的深度.
    方法
    find_path(T)
        给定一棵树，它会找到单个数据点的路径.

    """

    def __init__(self, x, itree):
        self.x = x
        self.e = 0
        self.path = self.find_path(itree.root)

    def find_path(self, T):
        # 给定一棵树，根据存储在每个节点上的拆分条件找到单个数据点的路径。
        if T.ntype == 'exNode':
            if T.size <= 1:
                return self.e
            else:
                T_len = 2.0 * (np.log(T.size - 1) + 0.5772156649) - (2.0 * (T.size - 1.) / (T.size * 1.0))
                self.e = self.e + T_len
                return self.e
        else:
            p = T.p  # 截取用于在给定节点上拆分数据的超平面
            n = T.n  # 用于在给定节点拆分数据的超平面的法向量
            self.e += 1
            if (self.x - p).dot(n) < 0:
                return self.find_path(T.left)
            else:
                return self.find_path(T.right)

