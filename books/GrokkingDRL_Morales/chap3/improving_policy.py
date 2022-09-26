import gym
import random
import numpy as np


def generate_random_policy(env):
    action_space_size = env.action_space.n
    observation_space_size = env.observation_space.n

    pi = [env.action_space.sample() for _ in range(observation_space_size)]

    return pi
    
    
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    """ 
    Function that evaluate policy: we estimate the reward for each state
    """
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros(len(P))

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi[s]]:
                #  print(prob, next_state, reward, done)
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(prev_V - V)) < theta: # check if new policy has converged
            break

        prev_V = V.copy()

    return V
    

def policy_improvement(V, P, gamma=1.0):
    """ 
    Function that improve policy by performing a one step look ahead
    """
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = [a for s, a in enumerate(np.argmax(Q, axis=1))]

    return new_pi

def policy_iteration(env, gamma=1.0, theta=1e-10):
    """ 

    """
    pi = generate_random_policy(env)
    P = env.env.P

    while True:
        old_pi = pi.copy()
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)

        if old_pi == pi:
            break

    return V, pi
    
def value_iteration(P, gamma=1.0, theta=1e-10):
    """ 

    """
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
    
        V = np.max(Q, axis=1)

    pi = [a for s, a in enumerate(np.argmax(Q, axis=1))]

    return V, pi
        


def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    P = env.env.P

    # Method 1: Policy iteration
    V, pi = policy_iteration(env)
    print(V, pi)

    # Method 2: Value Iteration
    V, pi = value_iteration(P)
    print(V, pi)
    


if __name__ == "__main__":
    main()

