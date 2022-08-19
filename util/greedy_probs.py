import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """ある状態Sの時の方策をQ関数をε-greedyにすることで得る
    """
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)  # 全体の確率が1になるように
    return action_probs
