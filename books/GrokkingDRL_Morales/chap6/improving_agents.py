import gym
import numpy as np
from tqdm import tqdm


def decay_schedule(init_value, min_value, decay_ratio, max_steps, 
        log_start=-2, log_base=-10):
    decay_steps = int(decay_ratio * max_steps)
    remaining_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, remaining_steps), 'edge')
    return values
    


def generate_trajectories(select_action, Q, epsilon, env, max_steps=200):
    trajectory, done = [], False

    while not done: 
        state = env.reset()

        for t in range(max_steps):
            action = select_action(state, Q, epsilon)
            next_state, reward, done, _ = env.step(action)
            #  experience = (state, action, reward, next_state, done)
            experience = [state, action, reward, next_state, done]
            trajectory.append(experience)
            if done: 
                break
            # if trajectory takes more than max steps to finish, generate a new one
            if t > max_steps: 
                trajectory = []
                break
            state = next_state

    #  return np.array(trajectory)
    return trajectory

    
def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, 
        alpha_decay_ratio=0.1, init_epsilon=0.1, min_epsilon=0.05, 
        epsilon_decay_ratio=0.01, n_episodes=1000, max_steps=200, 
        first_visit=True):
    # init
    nS, nA = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, 
                             n_episodes)
    pi_track = []
    Q = np.zeros(shape=(nS, nA), dtype=np.float64)
    Q_track = np.zeros(shape=(n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    # 
    for e in tqdm(range(n_episodes)):
        trajectory = generate_trajectories(select_action, Q, epsilon[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=bool)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue

            visited[state][action] = True
            n_steps = len(trajectory[t:])
            trajectory_rewards = [episode[2] for episode in trajectory[t:]]
            G = np.sum(discounts[:n_steps] * trajectory_rewards)
            Q[state][action] += alphas[e] * (G - Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = [a for s, a in enumerate(np.argmax(Q, axis=1))]

    return Q, V, pi, Q_track, pi_track




def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    Q, V, pi, Q_track, pi_track = mc_control(env)
    

if __name__ == "__main__":
    main()
