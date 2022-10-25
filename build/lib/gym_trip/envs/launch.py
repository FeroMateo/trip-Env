import gym

from stable_baselines3 import DQN

env = gym.make("gym_trip:tripEnv-v0")

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_trip_tensorboard/",learning_starts=12000)
model.learn(total_timesteps=500, log_interval=5)
print("Se guardo Trip")
model.save("dqn_trip_500k")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_trip_500k")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
