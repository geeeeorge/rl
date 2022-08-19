from collections import defaultdict, deque
import numpy as np

from env.gridworld import GridWorld
from utils.greedy_probs import greedy_probs


class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  #  ターゲット方策
        self.b = defaultdict(lambda: random_actions)  # 挙動方策
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)  # SARSAを保持

    def get_action(self, state):
        action_probs = self.b[state]  # 挙動方策からサンプリング
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            # 方策オフ型 -> 方策重点サンプリング
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho * (reward + self.gamma * next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策改善
        # ターゲット方策はgreedy
        self.pi[state] = greedy_probs(self.Q, state, 0)
        # 挙動方策はε-greedy
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == '__main__':
    env = GridWorld()
    agent = SarsaOffPolicyAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)
            if done:
                agent.update(next_state, None, None, None)  # 最後ゴールするところの学習のため
                break
            state = next_state
    
    env.render_q(agent.Q)
