import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random    # 随机策略时用到


def str_key(*args):
    '''
    将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)


def set_dict(target_dict, value, *args):
    '''
    Q(s,a) = value
    :param target_dict: Q
    :param value: value
    :param args: s_a
    '''
    target_dict[str_key(*args)] = value


def get_dict(target_dict, *args):
    '''
    value = Q(s,a)
    :param target_dict: Q
    :param args: s_a
    :return: value
    '''

    return target_dict.get(str_key(*args), 0)


def greedy_pi(A, s, Q, a):
    '''
    依据贪婪选择，计算在行为空间A中，状态s下，a行为被贪婪选中的几率
    考虑多个行为的价值相同的情况
    eg：4个最大Q值为0.25
    :param A: 行为空间A
    :param s: 状态s
    :param Q: Q表
    :param a: 行为a
    :return: 概率probability
    '''
    # max_q 为最大Q值 a_max_q 为最大Q值对应的a
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:  # 统计后续状态的最大价值以及到达到达该状态的行为（可能不止一个）
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:  # 多个最大Q值情况
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0:
        return 0.0
    return 1.0/n if a in a_max_q else 0.0


def greedy_policy(A, s, Q):
    '''
    在给定一个状态下，从行为空间A中选择一个行为a，使得Q(s,a) = max(Q(s,))，考虑到多个行为价值相同的情况
    :param A: 行为空间A
    :param s: 状态s
    :param Q: Q表
    :return: 某一个最大Q值的行为a
    '''

    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)

    return random.choice(a_max_q)


def epsilon_greedy_pi(A, s, Q, a, epsilon=0.1):
    '''
    ε-greedy π（a|s）={ ε/m + 1-ε * probability(考虑多个最大Q值情况) if a* = argmax Q(s,a) (a ∈ A)
                     |
                     { ε/m      else
    :param A: 行为空间A
    :param s: 状态s
    :param Q: Q表
    :param a: 行为a
    :param epsilon: ε的值
    :return: ε-greedy π（a|s）概率
    '''
    m = len(A)
    greedy_p = greedy_pi(A, s, Q, a)
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0/greedy_p)
    return (1 - epsilon) * greedy_p + epsilon/m


def epsilon_greedy_policy(A, s, Q, epsilon, show_random_num=False):
    '''
    采取ε-greedy的行为(集)
    :param A: 行为空间A
    :param s: 状态s
    :param Q: Q表
    :param epsilon: ε
    :param show_random_num:是否显示随机数拟
    :return: ε-greedy的行为(集)
    '''
    pis = []  # 所有行为的概率
    m = len(A)
    for i in range(m):
        pis.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    rand_value = random.random()  # 产生一个【0,1】的随机数

    for i in range(m):
        if show_random_num:
            print("随机数:{:.2f}, 拟减去概率{}".format(rand_value, pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:
            return A[i]
        
        
def draw_value(value_dict, useable_ace = True, is_q_dict = False, A = None):
    # 定义figure
    fig = plt.figure()
    # 将figure变为3d
    ax = Axes3D(fig)
    # 定义x, y
    x = np.arange(1, 11, 1) # 庄家第一张牌
    y = np.arange(12, 22, 1) # 玩家总分数
    # 生成网格数据
    X, Y = np.meshgrid(x, y)
    # 从V字典检索Z轴的高度
    row, col = X.shape
    Z = np.zeros((row,col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i,j])+"_"+str(Y[i,j])+"_"+str(useable_ace)
            if not is_q_dict:
                Z[i,j] = get_dict(value_dict, state_name)
            else:
                assert(A is not None)
                for a in A:
                    new_state_name = state_name + "_" + str(a)
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i,j]:
                        Z[i,j] = q
    # 绘制3D曲面
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, color="lightgray")
    plt.show()
    

def draw_policy(policy, A, Q, epsilon, useable_ace = False):
    def value_of(a):
        if a == A[0]:
            return 0
        else:
            return 1
    rows, cols = 11, 10
    useable_ace = bool(useable_ace)
    Z = np.zeros((rows, cols))
    dealer_first_card = np.arange(1, 12) # 庄家第一张牌
    player_points = np.arange(12,22)
    for i in range(11, 22): # 玩家总牌点
        for j in range(1, 11): # 庄家第一张牌 
            s = j, i, useable_ace
            s = str_key(s)
            a = policy(A, s, Q, epsilon)
            Z[i-11,j-1] = value_of(a)
            #print(s, a)
    
    plt.imshow(Z, cmap=plt.cm.cool, interpolation=None, origin="lower", extent=[0.5,11.5,10.5,21.5])