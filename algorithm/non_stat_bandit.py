import numpy as np

from algorithm.bandit import Bandit, Agent


class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        # ε-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8
    all_rates = np.zeros((runs, steps))
    non_stat_all_rates = np.zeros((runs, steps))

    for run in range(runs):
        # 定常のバンディット
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        # 非定常のバンディット
        non_stat_bandit = NonStatBandit()
        non_stat_agent = AlphaAgent(epsilon, alpha)
        non_stat_total_reward = 0
        non_stat_rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step+1))

            action = non_stat_agent.get_action()
            reward = non_stat_bandit.play(action)
            non_stat_agent.update(action, reward)
            non_stat_total_reward += reward
            non_stat_rates.append(non_stat_total_reward / (step+1))
        
        all_rates[run] = rates
        non_stat_all_rates[run] = non_stat_rates
    avg_rates = np.average(all_rates, axis=0)
    avg_non_stat_rates = np.average(non_stat_all_rates, axis=0)

    # draw graph
    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(avg_rates, label='sample average')
    plt.plot(avg_non_stat_rates, label='alpha const update')
    plt.legend()
    plt.show()