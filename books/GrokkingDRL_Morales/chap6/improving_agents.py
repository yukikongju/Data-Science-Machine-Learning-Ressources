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


def select_action(state, Q, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(len(Q[state]))
    return action


def sarsa(env, gamma=1.0, init_alpha = 0.5, alpha_min = 0.01, 
          alpha_decay_ratio=0.05, init_epsilon=0.3, epsilon_min=0.01, 
          epsilon_decay_ratio=0.05, n_episodes=1000, max_steps=200):
    # initialization
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    alphas = decay_schedule(init_alpha, alpha_min, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, epsilon_min, epsilon_decay_ratio, n_episodes)
    pi_track = np.zeros((n_episodes, nS), dtype=int)

    for e in tqdm(range(n_episodes)):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])
        while not done: 
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alphas[e] * td_error
            state, action = next_state, next_action
        Q_track[e] = Q
        pi_track[e] = np.argmax(Q, axis=1)

    V = np.max(Q, axis=1)
    pi = {s:a for s,a in enumerate(np.argmax(Q, axis=1))}

    return Q, V, pi, Q_track, pi_track

def q_learning(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.05,
        init_epsilon=0.4, min_epsilon=0.05, epsilon_decay_ratio=0.05, 
        n_episodes=1000, max_steps=200):
    # initilization
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros(shape=(nS, nA), dtype=np.float64)
    Q_track = np.zeros(shape=(n_episodes, nS, nA), dtype=np.float64)
    pi_track = np.zeros((n_episodes, nS), dtype=np.int8)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    #
    for e in tqdm(range(n_episodes)):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done: 
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            qtarget = reward + gamma * np.max(Q[next_state])
            qerror = qtarget - Q[state][action]
            Q[state][action] += alphas[e] * qerror
            state, action = next_state, next_action

        Q_track[e] = Q
        pi_track[e] = np.argmax(Q, axis=1)

    V = np.max(Q, axis=1)
    pi = {s:a for s,a in enumerate(np.argmax(Q, axis=1))}

    return Q, V, pi, Q_track, pi_track



def main():
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    #  Q, V, pi, Q_track, pi_track = mc_control(env)
    #  Q, V, pi, Q_track, pi_track = sarsa(env)
    Q, V, pi, Q_track, pi_track = q_learning(env)
    breakpoint()
    

if __name__ == "__main__":
    main()
