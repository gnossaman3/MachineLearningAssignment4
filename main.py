import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        #self.fc3 = nn.Linear(100, 100)
        #self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def Frozen_Lake_Experiments():
    # 0 = left; 1 = down; 2 = right;  3 = up

    environment = 'FrozenLake-v0'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc

    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10

    ### POLICY ITERATION ####
    print('POLICY ITERATION WITH FROZEN LAKE')
    for i in range(0, 10):
        st = time.time()
        best_policy, k = policy_iteration(env, gamma=(i + 0.5) / 10)
        scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / 10)
        end = time.time()
        gamma_arr[i] = (i + 0.5) / 10
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake Policy Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(fname='FozenPolicyTime')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake Policy Iteration Reward Analysis')
    plt.grid()
    plt.savefig(fname='FrozenPolicyReward')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Frozen Lake Policy Iteration Convergence Analysis')
    plt.grid()
    plt.savefig(fname='FrozenPolicyConvergence')
    plt.close()

    ### VALUE ITERATION ###
    print('VALUE ITERATION WITH FROZEN LAKE')
    best_vals = [0] * 10
    for i in range(0, 10):
        st = time.time()
        best_value, k = value_iteration(env, gamma=(i + 0.5) / 10)
        policy = extract_policy(env, best_value, gamma=(i + 0.5) / 10)
        policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
        end = time.time()
        gamma_arr[i] = (i + 0.5) / 10
        iters[i] = k
        best_vals[i] = best_value
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Frozen Lake Value Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(fname='FrozenValueTime')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Frozen Lake Value Iteration Reward Analysis')
    plt.grid()
    plt.savefig(fname='FrozenValueReward')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Frozen Lake Value Iteration Convergence Analysis')
    plt.grid()
    plt.savefig(fname='FrozenValueConvergence')
    plt.close()

    plt.plot(gamma_arr, best_vals)
    plt.xlabel('Gammas')
    plt.ylabel('Optimal Value')
    plt.title('Frozen Lake Value Iteration Best Value Analysis')
    plt.grid()
    plt.savefig(fname='FrozenValueBest')
    plt.close()

    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []
    time_array = []
    Q_array = []
    for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.95
        episodes = 30000
        environment = 'FrozenLake-v0'
        env = gym.make(environment)
        env = env.unwrapped
        #desc = env.unwrapped.desc
        for episode in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 1000000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < (epsilon):
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsilon = (1 - 2.71 ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time.time()
        # print("time :",end-st)
        time_array.append(end - st)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)

    plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0], label='epsilon=0.05')
    plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1], label='epsilon=0.15')
    plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.25')
    plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsilon=0.50')
    plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4], label='epsilon=0.75')
    plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsilon=0.95')
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Frozen Lake Q Learning Constant Epsilon')
    plt.ylabel('Average Reward')
    plt.savefig(fname='FrozenQRewards')
    plt.close()

    plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], time_array)
    plt.xlabel('Epsilon Values')
    plt.grid()
    plt.title('Frozen Lake Q Learning Execution Times')
    plt.ylabel('Execution Time (s)')
    plt.savefig(fname='FrozenQTimes')
    plt.close()

    plt.subplot(1, 6, 1)
    plt.imshow(Q_array[0])
    plt.title('e=0.05')

    plt.subplot(1, 6, 2)
    plt.title('e=0.15')
    plt.imshow(Q_array[1])

    plt.subplot(1, 6, 3)
    plt.title('e=0.25')
    plt.imshow(Q_array[2])

    plt.subplot(1, 6, 4)
    plt.title('e=0.50')
    plt.imshow(Q_array[3])

    plt.subplot(1, 6, 5)
    plt.title('e=0.75')
    plt.imshow(Q_array[4])

    plt.subplot(1, 6, 6)
    plt.title('e=0.95')
    plt.imshow(Q_array[5])
    plt.colorbar()


    plt.savefig(fname='FrozenQArray')
    plt.close()

def LunarLander_Experiments():
    environment = 'LunarLander-v2'
    env = gym.make(environment)
    env = env.unwrapped
    net = Net().float()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10

    ### POLICY ITERATION TAXI: ####
    print('POLICY ITERATION WITH LANDER')

    for i in range(3, 10):
        st = time.time()
        best_policy, k = policy_iteration(env, gamma=(i + 0.5) / 10)
        scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / 10)
        end = time.time()
        gamma_arr[i] = (i + 0.5) / 10
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st



    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Lander Policy Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(fname='LanderPolicyTime')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Lander Policy Iteration Reward Analysis')
    plt.grid()
    plt.savefig(fname='LanderPolicyReward')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Lander - Policy Iteration - Convergence Analysis')
    plt.grid()
    plt.savefig(fname='LanderPolicyConvergence')
    plt.close()


    #### VALUE ITERATION LANDER: #####
    print('VALUE ITERATION WITH LANDER')
    best_vals = [0.0] * 10
    for i in range(2, 8):
        print(i)
        st = time.time()
        best_value, k = value_iteration(env, gamma=(i + 0.5) / 10);
        policy = extract_policy(env, best_value, gamma=(i + 0.5) / 10)
        policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
        end = time.time()
        gamma_arr[i] = (i + 0.5) / 10
        iters[i] = k
        time_array[i] = end - st
        list_scores[i] = np.mean(policy_score)
        best_vals[i] = np.mean(best_value)
        print(policy)

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Lander Value Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(fname='LanderValueTime')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Lander Value Iteration Reward Analysis')
    plt.grid()
    plt.savefig(fname='LanderValueReward')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Lander Value Iteration Convergence Analysis')
    plt.grid()
    plt.savefig(fname='LanderValueConvergence')
    plt.close()


    plt.plot(gamma_arr, best_vals)
    plt.xlabel('Gammas')
    plt.ylabel('Optimal Value')
    plt.title('Lander Value Iteration Best Value Analysis')
    plt.grid()
    plt.savefig(fname='LanderValueBest')
    plt.close()


    print('Q LEARNING WITH LANDER')
    st = time.time()
    rewards = []
    iters = []
    alpha = 1.0
    gamma = 1.0
    episodes = 20000
    epsilon = 1.0
    times = []
    episode_time_start = time.time()
    for episode in range(episodes):
        if episode % 100 == 0:
            print('Episode ', episode)
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = 100000
        for i in range(max_steps):
            if done:
                break
            current = state
            Q = net(torch.from_numpy(current).float().view(-1, 8))
            Q_new = torch.clone(Q)
            if np.random.rand() < epsilon:
                #action = np.argmax(Q[current, :])
                action = torch.argmax(net(torch.from_numpy(current).float().view(-1,8)), axis=1)
            else:
                action = env.action_space.sample()
            state, reward, done, info = env.step(int(action))
            t_reward += reward
            temp = alpha * (reward + gamma * float(torch.max(net(torch.from_numpy(state).float().view(-1,8))[0, :])) - float(net(torch.from_numpy(current).float().view(-1,8))[0, action]))
            Q_new[0, action] += temp
            loss = criterion(Q_new, Q)
            loss.backward()
            optimizer.step()
        episode_time_end = time.time()
        epsilon *= 0.99
        alpha = 2.71 ** (-episode / 1000)
        rewards.append(t_reward)
        times.append(episode_time_end-episode_time_start)
        iters.append(i)

    print("average :", np.average(rewards[2000:]))
    env.close()
    end = time.time()
    print("time :", end - st)

    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    size = 5
    chunks = list(chunk_list(rewards, size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]
    plt.plot(range(0, len(rewards), size)[200:], averages[200:])
    plt.xlabel('iters')
    plt.ylabel('Average Reward')
    plt.title('Lander Q Learning Rewards')
    plt.savefig(fname='LanderQRewards')
    plt.close()

    plt.plot(range(0, len(times)), times)
    plt.xlabel('iters')
    plt.ylabel('Accumulated Time (s)')
    plt.title('Lander Q Learning Execution Times')
    plt.savefig(fname='LanderQTimes')
    plt.close()


def run_episode(env, policy, gamma, render=True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[np.argmax(obs)]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma):
    try:
        num_space = env.nS
        num_action = env.nA
        policy = np.zeros(num_space)
        for s in range(num_space):
            q_sa = np.zeros(num_action)
            for a in range(num_action):
                q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
            policy[s] = np.argmax(q_sa)
        return policy
    except:
        num_space = 8
        num_action = env.action_space.n
        policy = np.zeros(num_space)
        for s in range(num_space):
            q_sa = np.zeros(num_action)
            for a in range(num_action):
                s_, r, is_done, info = env.step(a)
                q_sa[a] = (r + gamma * v[np.argmax(s_)])
            policy[s] = np.argmax(q_sa)
        return policy


def compute_policy_v(env, policy, gamma):
    try:
        num_space = env.nS
        num_action = env.nA
        v = np.zeros(num_space)
        eps = 1e-5
        while True:
            prev_v = np.copy(v)
            for s in range(num_space):
                policy_a = policy[s]
                v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.step(policy_a)])
            if (np.sum((np.fabs(prev_v - v))) <= eps):
                break
    except:
        num_space = 8
        num_action = env.action_space.n
        v = np.zeros(num_space)
        eps = 1e-5
        while True:
            prev_v = np.copy(v)
            for s in range(num_space):
                policy_a = policy[s]
                s_, r, is_done, info = env.step(int(policy_a))
                v[s] = r + gamma * prev_v[np.argmax(s_)]
            if (np.sum((np.fabs(prev_v - v))) <= eps):
                break
    return v


def policy_iteration(env, gamma):
    try:
        num_space = env.nS
        num_action = env.nA
    except:
        num_action = env.action_space.n
        num_space = 8
    policy = np.random.choice(num_action, size=(num_space))
    max_iters = 200000
    try:
        desc = env.unwrapped.desc
    except:
        pass
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k = i + 1
            break
        policy = new_policy
    return policy, k


def value_iteration(env, gamma):
    try:
        num_space = env.nS
        num_action = env.nA
        v = np.zeros(num_space)  # initialize value-function
        max_iters = 100000
        eps = 1e-20
        try:
            desc = env.unwrapped.desc
        except:
            pass
        for i in range(max_iters):
            prev_v = np.copy(v)
            for s in range(num_space):
                q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(num_action)]
                v[s] = max(q_sa)
            # if i % 50 == 0:
            #	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())
            if (np.sum(np.fabs(prev_v - v)) <= eps):
                k = i + 1
                break
        return v, k
    except:
        num_space = 8
        num_action = env.action_space.n
        v = np.zeros(num_space)  # initialize value-function
        max_iters = 100000
        eps = 1e-20
        try:
            desc = env.unwrapped.desc
            for i in range(max_iters):
                prev_v = np.copy(v)
                for s in range(num_space):
                    q_sa = [sum((r + gamma * prev_v[s_]) for s_, r, is_done, info in env.step(a)) for a in
                            range(num_action)]
                    v[s] = max(q_sa)
                # if i % 50 == 0:
                #	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())
                if (np.sum(np.fabs(prev_v - v)) <= eps):
                    k = i + 1
                    break
        except:
            for i in range(max_iters):
                prev_v = np.copy(v)
                for s in range(num_space):
                    q_sa = []
                    for j in range(num_action):
                        s_, r, is_done, info = env.step(j)
                        q_sa.append(r + gamma * prev_v[np.argmax(s_)])
                    v[s] = max(q_sa)
                if (np.sum(np.fabs(prev_v - v)) <= eps):
                    k = i + 1
                    break
        return v, k



def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.savefig(title + str('.png'))
    plt.close()

    return (plt)


print('STARTING EXPERIMENTS')
Frozen_Lake_Experiments()
LunarLander_Experiments()
print('END OF EXPERIMENTS')
