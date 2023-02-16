import networkx as nx
import numpy as np

# @author: Chris 2023/02/15
# Learn node2vec


# 总结：输出 每个节点较小维度的特征向量
# 1. 对Graph进行抽样  l * r * n  n 每个节点随机游走 l长度序列 表示节点在图中与其他节点的关系 r次尽可能消除单词游走产生的偏差
# 2. 得到的 l * r * n 的图语言序列 输入到 word2vec （设定 特征向量长度 k）
# 3. 输出 n * k的矩阵

G = nx.les_miserables_graph()


def init_transition_prob(self):
    """
    :param self:
    :return: 归一化的转移矩阵
    """
    g = self.G
    nodes_info, edges_info = {}, {}  # 字典
    for node in g.nodes:
        # 因为 probs中是没有存节点 id
        nbrs = sorted(g.neighbors(node))  # 当前节点的邻居节点 标签 排序
        probs = [g[node][n]['weight'] for n in nbrs]  # 权重 当成 概率
        # 归一化
        norm = sum(probs)
        normalized_probs = [float(n) / norm for n in probs]
        nodes_info[node] = self.alias_set(normalized_probs)  # 通过别名抽样得到 accept 和 alia 表，为第二次抽样做准备

    for edge in g.edges:

        # 有向图
        if g.is_directed():
            edges_info[edge] = self.get_alias_edge(edge[0].edge[1])
        # 无向图
    # nodes_info 格式 {{节点：accep，alias}，.....}
    self.nodes_info = nodes_info

    # edges_info 格式 {{边：accept，alias}，......}
    self.edges_info = edges_info


def alias_setup(self, probs):
    """
    :probs: v到所有x的概率
    :return: Alias数组与Prob数组
    """
    K = len(probs)
    q = np.zeros(K)  # 对应Prob数组
    J = np.zeros(K, dtype=np.int)  # 对应Alias数组
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []  # 存储比1小的列
    larger = []  # 存储比1大的列
    for kk, prob in enumerate(probs):
        q[kk] = K * prob  # 概率
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.

    # 通过拼凑，将各个类别都凑为1
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large  # 填充Alias数组
        q[large] = q[large] - (1.0 - q[small])  # 将大的分到小的上

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def get_alias_edge(self, t, v):
    # 当前进行第 N 次 采样, 第 N-1 次采样节点为 v，第 N-2 次采样节点为 t
    # p 为bfs-like采样超参； q 为 dfs-like采样超惨

    g = self.G
    p = self.p
    q = self.q

    unnormalized_probs = []

    for v_nbr in sorted(g.neighbors(v)):

        # 如果 v_nbr 为 第 N-2 次采样节点 t，那么v_nbr的非正则化转移概率为: 权重 / p
        if v_nbr == t:
            unnormalized_probs.append(g[v][v_nbr]['weight'] / p)

        # 如果 v_nbt 为 第 N-2 次采样节点 t的邻接点，其非正则化转移概率为：权重
        elif g.has_edge(v_nbr, t):
            unnormalized_probs.append(g[v][v_nbr]['weight'])

        # 如果 v_nbr 为 第 N-1 次采样节点 v 的邻接点且不是t的邻接点，其非正则化转移概率为：权重 / q
        else:
            unnormalized_probs.append(g[v][v_nbr]['weight'] / q)

    # 正则化节点v的NBR采样概率
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return self.alias_setup(normalized_probs)


def alias_draw(self, J, q):
    """
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    """
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int((np.random.random() * K))  # 随机取一列

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.random() < q[kk]:  # 比较
        return kk
    else:
        return J[kk]


def node2vecWalk(self, u):
    # 起点为u 采样长度为 l
    # 也就是说 第 1 次采样为u
    # 现在从 第 2 次开始采样
    walk = [u]
    g = self.G
    l = self.l

    nodes_info, edges_info = self.nodes_info, self.edges_info

    while len(walk) < l:
        # curr 为倒数 第一个
        curr = walk[-1]

        v_curr = sorted(g.neighbors(curr))

        if len(v_curr) > 0:
            # 说明当前处于 第 2 次采样中，则情况只有一种：直接从curr_nbr中采样
            if len(walk) == 1:
                # nodes_info[curr][0] -> accept /// nodes_info[curr][1] -> alias
                accept = nodes_info[curr][0]
                alias = nodes_info[curr][1]
                walk.append(v_curr[self.alias_draw(accept, alias)])
            else:
                prev = walk[-2]
                accept = edges_info[(prev, curr)][0]
                alias = edges_info[(prev, curr)][1]
                ne = v_curr[self.alias_draw(accept, alias)]
                walk.append(ne)
        else:
            break


def learning_features(self):
    # r次抽样 每次都要产生n个长度为l的游走序列 r * n * l
    # test
    g = self.G
    walks = []
    nodes = list(g.nodes())
    for iter in range(self.r):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = self.node2vecWalk(node)
            walks.append(walk)
    # embedding

    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(sentences=walks, vector_size=self.d, window=self.k, min_count=0, sg=1, workers=3)

    # 就可以得到 每个节点的向量了 可用于 节点分类
    f = model.wv
    return f





