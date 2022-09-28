import gym
import numpy as np

from tqdm import tqdm

def softmax(env, init_temp=1000.0, min_temp=0.01, decay_ratio=0.04, n_episodes=5000):
    """ 
    Exploration-Explotation method using softmax
    """
    env.reset()

    # initialization
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    for e in tqdm(range(n_episodes)):
        # calculate temperature
        decay_episodes = n_episodes * decay_ratio
        temp = 1 - e / decay_episodes
        temp *= init_temp - min_temp
        temp += min_temp
        temp = np.clip(temp, min_temp, init_temp)

        # calculate policy
        scaled_Q = Q / temp
        normalized_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(normalized_Q)
        probs = exp_Q / np.sum(exp_Q)

        # select action
        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        _, reward, _, _ = env.step(action)

        # updates
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return returns, actions, Qe
    
def upper_confidence_bound(env, n_episodes=5000):
    pass
    

def thompson_sampling(env, n_episodes=5000):
    pass
    

def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    returns, actions, Qe = softmax(env)
    

if __name__ == "__main__":
    main()
