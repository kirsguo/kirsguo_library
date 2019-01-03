import gym
from puckworld_env import PuckWorldEnv
from ddpg_agent import DDPGAgent
from utils import learning_curve
import numpy as np

env = PuckWorldEnv()
agent =DDPGAgent(env)
data = agent.learning(max_episode_num=200, display=True)
learning_curve(data, 2, 1, #title="DDPGAgent performance on PuckWorld with continuous action space",
               x_name="episodes", y_name="rewards of episode")