import gym
import json
import datetime as dt
from sklearn import preprocessing

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('~/Downloads/HitBTC.csv')
df = df.sort_values('timestamp')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO('MlpPolicy', env, verbose=1,device="cpu")
model.learn(total_timesteps=40000)

obs = env.reset()
for i in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
