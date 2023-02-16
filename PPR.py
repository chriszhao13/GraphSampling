# 简化版的PR
import networkx
import numpy as np

# 总结： 要进行累计进行多次随机游走（但是感觉是前后依赖的）才能收敛 可以直接矩阵求解
def train(alpha, graph, root, iterations):

    rank = {x: 0 for x in graph.nodes}
    # 初始概率为 1
    rank[root] = 1
    count = 0

    while True:
        tmp = {x: 0 for x in graph.nodes}
        for node in graph.nodes:
            out_nodes = list(graph.neighbors(node))
            out_degree = len(out_nodes)
            for j in out_nodes:
                # 可不要
                data = graph.get_edge_data(node, j)
                # 当前节点i的邻接点j分到这么多权重 alpha * rank[node]
                tmp[j] += alpha * rank[node] / out_degree * data['weight']
        tmp[root] += (1 - alpha)
        rank = tmp
        count += 1
        if count >= iterations:
            # print('PersonalRank:%d' % count)
            break
    return rank


graph = networkx.DiGraph()
graph.add_edge('任小牛', '笔记本电脑', weight=1)
graph.add_edge('任小牛', '风扇', weight=0.1)
graph.add_edge('任小牛', '键盘', weight=0.1)
graph.add_edge('卡洛斯', '笔记本电脑', weight=0.2)
graph.add_edge('卡洛斯', '风扇', weight=0.3)
graph.add_edge('詹姆斯', '风扇', weight=0.4)
graph.add_edge('詹姆斯', '键盘', weight=0.5)
graph.add_edge('卡尔', '笔记本电脑', weight=0.7)
graph.add_edge('卡尔', '键盘', weight=0.9)

target = '任小牛'
rs = train(0.85,graph, target, 2000)
rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)
print(rs)
# # 另一个
# rs = rank.train_matrix(graph, target)
# rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)
# print(rs)
# # gmres
# rs = rank.train_csr_matrix(graph, target)
# rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)
# print(rs)
