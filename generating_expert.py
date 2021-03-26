import pybullet_envs
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from sac_torch import Agent

seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env_name = 'InvertedPendulumBulletEnv-v0'
env = gym.make(env_name)
if seed is not None:
    env.seed(seed)
state = env.reset()
# initializing a model
model = Agent(input_dims=env.observation_space.shape[0], env=env, 
                n_actions=env.action_space.shape[0])

mean_rewards = []
for i in range(100):
    rewards = [model.train_on_env(env) for _ in range(100)] 
    mean_rewards.append(np.mean(rewards))
    if i % 5:
        print("mean reward:%.3f" % (np.mean(rewards)))
        plt.figure(figsize=[9, 6])
        plt.title("Mean reward per 100 games")
        plt.plot(mean_rewards)
        plt.grid()
        # plt.show()
        plt.savefig('plots/PG_learning_curve.png')
        plt.close()
    
    if np.mean(rewards) > 1000:
        print("TRAINED!")
        break

model.save_models()
#model.load("experts/saved_expert/pg.model")

num_expert = 100

expert_samples = np.array([model.generate_session(env) for i in range(num_expert)])
np.save('expert_samples/pg_cartpole', expert_samples)