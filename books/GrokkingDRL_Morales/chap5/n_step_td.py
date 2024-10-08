import gym
import numpy as np
import matplotlib.pyplot as plt

from utils import get_action, decay_schedule, generate_trajectory, get_random_stochastic_policy
from tqdm import tqdm


def n_step_td(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, 
        alpha_decay_ratio=0.5, n_steps=3, n_episodes=500):
    """ 
    Evaluating state function with n steps td learning
    """
    # initialize variables
    nS = env.observation_space.n
    V = np.zeros(shape=(nS,), dtype=np.float64)
    V_track = np.zeros(shape=(n_episodes, nS), dtype=np.float64)

    discounts = np.logspace(0, n_steps+1, num=n_steps+1, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes)):
        # initialize variable for episode
        state, done, path = env.reset(), False, []

        # generate path
        while not done and len(path) < n_steps:
            action = get_action(pi, state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            path.append(experience)
            if done: 
                break

        # update state value function
        num_experiences = len(path)
        rewards = np.array(path)[:, 2]
        G = np.sum(discounts[:num_experiences] * rewards)
        nstep_error = G - V[state]
        V[state] = V[state] + alphas[e] * nstep_error
        V_track[e] = V

    return V, V_track
    

def forward_td(pi, env, lambda_= 0.95, gamma=1.0):
    """ 
    We are weighting each experience differently
    """
    pass
    

def backward_td(pi, env, gamma=1.0, init_alpha=0.05, min_alpha=0.01, 
        alpha_decay_ratio=0.5, lambda_=0.3, n_episodes=500):
    """ Decay Elligibility Traces """
    # initialize variables
    nS = env.observation_space.n
    V = np.zeros(shape=(nS,), dtype=np.float64)
    V_track = np.zeros(shape=(n_episodes, nS), dtype=np.float64)
    E = np.zeros(shape=(nS,), dtype=np.float64) # elligibility traces

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    #  discounts = np.logspace(0, n_steps +1, base=gamma, endpoint=False)


    for e in tqdm(range(n_episodes)):
        E.fill(0)
        state, done = env.reset(), False

        # generate trajectory
        while not done:
            action = get_action(pi, state)
            next_state, reward, done, _ = env.step(action)
            E[state] += 1

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error * E[state]
            E[state] = E[state] * gamma * lambda_

            state = next_state
        
        V_track[e] = V

    return V, V_track
    


def main():
    env = gym.make('FrozenLake-v1')
    pi = get_random_stochastic_policy(env)
    #  V, V_track = n_step_td(pi, env, gamma=0.99, n_steps=10)
    V, V_track = backward_td(pi, env, gamma=0.95)
    print(V)


if __name__ == "__main__":
    main()



