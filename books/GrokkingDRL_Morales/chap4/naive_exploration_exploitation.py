import gym
import numpy as np
import random

from tqdm import tqdm

def pure_exploitation(env, n_episodes=5000):
    """ 
    Return the returns and the actions taken for all n_episodes

    """
    env.reset()

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    for e in tqdm(range(n_episodes)):
        action = np.argmax(Q)
        _, reward, _, _ = env.step(action)

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q

        returns[e] = reward
        actions[e] = action

    return returns, actions, Qe

def pure_exploration(env, n_episodes=5000):
    """ 

    """
    env.reset()

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    for e in tqdm(range(n_episodes)):
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q

        actions[e] = action
        returns[e] = reward

    return returns, actions, Qe

def epsilon_greedy(env, epsilon=0.01, n_episodes=500):
    """ 

    """
    env.reset()

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    for e in tqdm(range(n_episodes)):
        if np.random.random() > epsilon: 
            action = np.argmax(Q)
        else: 
            action = env.action_space.sample()

        _, reward, _, _ = env.step(action)

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q

        actions[e] = action
        returns[e] = reward

    return returns, actions, Qe


def linearly_decaying_epsilon_greedy(env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.05, n_episodes=5000):
    """ 
    EPSILON_DELTA = (EPSILON - MINIMUM_EPSILON)/STEPS_TO_TAKE
    """
    env.reset()

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    for e in tqdm(range(n_episodes)):
        # calculate epsilon
        decay_episodes = n_episodes * decay_ratio
        epsilon = 1 - e / decay_episodes
        epsilon *= init_epsilon - min_epsilon
        epsilon += min_epsilon
        epsilon = np.clip(epsilon, min_epsilon, init_epsilon)

        # choose action
        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = env.action_space.sample()

        # get reward from environment
        _, reward, _, _ = env.step(action)

        # calculate action-value function
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q

        # update returns and action history
        returns[e] = reward
        actions[e] = action

    return returns, actions, Qe


def exponentially_decaying_epsilon_greedy(env, init_epsilon=1.0, min_epsilon = 0.01, decay_ratio=0.05, n_episodes=5000):
    # initialization
    env.reset()

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int32)

    # calculate epsilons
    decay_episodes = int(n_episodes * decay_ratio)
    remaining_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, remaining_episodes))

    for e in tqdm(range(n_episodes)):

        # get action
        if np.random.random() > epsilons[e]:
            action = np.argmax(Q)
        else:
            action = env.action_space.sample()

        # get reward
        _, reward, _, _ = env.step(action)

        # update states
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]/N[action])
        Qe[e] = Q

        returns[e] = reward
        actions[e] = action

    return returns, actions, Qe


def optimistic_initialization(env, optimistic_estimate=1.0, initial_count=100, n_episodes=5000):
    """ 
    Select the action with the most uncertainty
    """
    pass
    


def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    #  returns, actions, Qe = pure_exploitation(env)
    #  returns, actions, Qe = pure_exploration(env)
    #  returns, actions, Qe = epsilon_greedy(env)
    #  returns, actions, Qe = linearly_decaying_epsilon_greedy(env)
    #  returns, actions, Qe = exponentially_decaying_epsilon_greedy(env)
    returns, actions, Qe = optimistic_initialization(env)


    

if __name__ == "__main__":
    main()


