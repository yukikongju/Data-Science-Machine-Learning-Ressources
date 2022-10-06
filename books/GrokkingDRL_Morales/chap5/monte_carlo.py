import gym
import numpy as np
from tqdm import tqdm


def get_random_deterministic_policy(env):
    """ 
    generate stochastic policy. size: observation_space x action_space
    """
    pi = np.random.randint(0, env.action_space.n, size=(env.observation_space.n), 
            dtype=np.int8)

    return pi


def get_random_stochastic_policy(env):
    """ 
    generate stochastic policy. size: observation_space x action_space
    """
    pi = np.random.rand(env.observation_space.n, env.action_space.n)
    pi /= pi.sum(axis=1, keepdims=1)
    return pi


def get_action(pi, state, epsilon=0.10):
    """ 
    We follow a epsilon-greedy approach when selecting our action
    """
    if np.random.rand() < epsilon:
        action = np.random.choice(range(len(pi[0])))
    else:
        action = np.argmax(pi[state])
    return action


def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, 
        log_base=-10):
    """ 
    exponentially decaying schedule
    """
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')

    return values
    

def generate_trajectory(pi, env, max_steps=20):
    """ 
    Get trajectory: list of experiences
    """
    state = env.reset()
    done, trajectory = False, []
    while not done:
        action = get_action(pi, state)
        next_state, reward, done, _ = env.step(action)
        #  env.render()

        experience = (state, action, reward, next_state, done)
        trajectory.append(experience)
        state = next_state

    return np.array(trajectory, np.object0)


def mc_prediction(pi, env, gamma=1.0, init_alpha=0.05, min_alpha=0.01, 
        alpha_decay_ratio=0.3, n_episodes=500, max_steps=100, first_visit=True):
    """ 
    Evaluating state-value function with monte carlo
    """
    # initialize variables
    nS = env.observation_space.n
    V = np.zeros((nS,), dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)

    # calculating offline variables
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma,
            endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, max_steps)


    # evaluating state
    for e in tqdm(range(n_episodes)):

        trajectory = generate_trajectory(pi, env, max_steps)
        visited = np.zeros(shape=(nS,), dtype=np.bool8)

        # calculate reward
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            V[state] = V[state] + alphas[t] * (G - V[state])

        V_track[e] = V

    return V.copy(), V_track


def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    env.reset()
    pi = get_random_stochastic_policy(env)
    V, V_track = mc_prediction(pi, env)
    print(V)


if __name__ == "__main__":
    main()

