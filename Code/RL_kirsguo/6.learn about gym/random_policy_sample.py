import gym
from gym import Env
from gridworld import WindyGridWorld  # 导入自建的有风世界环境
from core import Agent  # 导入自建的agent基类
from utils import learning_curve # 画图方法

env =WindyGridWorld()
env.reset()
env.render()

agent = Agent(env, capacity=10000)# 一个记忆容量为10000的agent
data = agent.learning(max_episode_num = 180, display = True)
learning_curve(data, 2, 0, title = "learning curve", x_name = "episode", y_name="time steps")
env.close()