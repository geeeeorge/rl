from collections import defaultdict
from env.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    """反復方策評価の1ステップ，全状態の価値関数を1回ずつ更新
    """
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    """反復方策評価
    """
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        # 更新された価値関数の値の最大値と閾値を比べる
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)