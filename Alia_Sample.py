import random

import numpy as np

# @author: Chris 2023/02/14
# 总结： 1. 别名表 2. 两次投掷
# 1. Get PDF Table
prob_num = 4
transform_prob = []
for i in range(prob_num):
    transform_prob.append(random.random())
print("归一化前： ", transform_prob)
norm_const = sum(transform_prob)

normalized_prob = [round(float(u_prob) / norm_const, 2) for u_prob in transform_prob]

print("归一化后：", normalized_prob, sum(normalized_prob))

# 2. *N and bucket in Big or small

length = prob_num

accept, alias = [0] * length, [0] * length

small, big = [], []

transform_N = np.array(normalized_prob) * length

print("归一化 * N 后：", transform_N)

for i, pro in enumerate(transform_N):
    if pro < 1.0:
        small.append(i)
    else:
        big.append(i)

print("small: ", small)
print("big: ", big)

# 3. 补齐

print("Start_________________________")
i = 0
while small and big:
    # 各出一个元素
    i = i + 1
    print("**************************")
    print("epoch : ", i)
    small_idx, large_idx = small.pop(), big.pop()
    # 小的 概率为 prob对应位置的 值
    accept[small_idx] = transform_N[small_idx]
    # 大的 索引放入
    alias[small_idx] = large_idx
    # 大的补 小的
    transform_N[large_idx] = transform_N[large_idx] - (1 - transform_N[small_idx])

    # 如果大的还是大于1 就放回大的 小于1 就放到小的

    if transform_N[large_idx] < 1.:
        small.append(large_idx)
    else:
        big.append(large_idx)
    print("small_id: ", small_idx)
    print("large_id: ", large_idx)
    print("accept: ", accept)
    print("alias: ", alias)
    print("transpose: ", transform_N)

while big:
    large_idx = big.pop()
    accept[large_idx] = 1
while small:
    small_idx = small.pop()
    accept[small_idx] = 1


def alias_sample(accept, alias):
    N = len(accept)
    # 生成随机索引 整数 1 到 N
    i = int(np.random.random() * N)
    # 生成0-1随机数
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


# 模拟 10000 次
# 结果抽样分布 近似 原始分布！！！！！
all_sample = {}

for i in range(10000):
    sample = alias_sample(accept, alias)
    if sample not in all_sample:
        all_sample[sample] = 0
    all_sample[sample] += 1

k = list(range(4))

v = [all_sample[key]  for key in k]
x = np.array(v) / sum(v)
import matplotlib.pyplot as plt

plt.title("PDF")
plt.bar(k, x)
plt.show()
