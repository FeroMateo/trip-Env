import gym

from stable_baselines3 import DQN

env = gym.make("gym_trip:tripEnv-v0")


model = DQN.load("dqn_trip_500k")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
