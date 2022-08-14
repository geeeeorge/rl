from collections import defaultdict
from gridworld import GridWorld

from policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma):
    """価値反復法の1ステップ，
    """
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_values = []

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    """価値反復法
    """
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

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
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma, is_render=True)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)