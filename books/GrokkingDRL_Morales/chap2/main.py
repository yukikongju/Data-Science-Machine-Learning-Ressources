import gym

env_name = "FrozenLake-v1"
env_name = "FrozenLake8x8-v1"
env = gym.make(env_name) 

print(env.P)
print(env.observation_space.n)
print(env.action_space.n)


