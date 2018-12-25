import gym
from puckworld import PuckWorldEnv
from agents import DQNAgent
from utils import learning_curve

env = PuckWorldEnv()
agent = DQNAgent(env)
data = agent.learning(gamma=0.99, epsilon=1,decaying_epsilon=True,alpha=1e-3,max_episode_num=100,display=False)
learning_curve(data, 2, 1, title="DQNAgent performance on PuckWorld",
               x_name="episodes", y_name="rewards of episode")