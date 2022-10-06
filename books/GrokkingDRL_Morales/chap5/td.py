import gym
import numpy as np

from tqdm import tqdm
from utils import get_random_stochastic_policy, get_action, decay_schedule, generate_trajectory


def td_prediction(pi, env, gamma=1.0, init_alpha=0.05, min_alpha=0.01, 
        alpha_decay_ratio=0.3, n_episodes=500, max_steps=100):
    """ 
    Evaluating state value function using temporal difference prediction
    """
    # initialize variables and offline calculs
    nS = env.observation_space.n
    V = np.zeros(shape=(nS,), dtype=np.float64)
    V_track = np.zeros(shape=(n_episodes, nS), dtype=np.float64)

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes)):

        state, done = env.reset(), False

        while not done:
            action = get_action(pi, state)
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error

            state = next_state

        V_track[e] = V

    return V, V_track
    


def main():
    env = gym.make('FrozenLake-v1')
    pi = get_random_stochastic_policy(env)
    V, V_track = td_prediction(pi, env)
    print(V)
    


if __name__ == "__main__":
    main()
