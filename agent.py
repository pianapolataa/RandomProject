import numpy as np

class QAgent:
    def __init__(self, n_actions=100, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.n_actions = n_actions
        self.q_table = np.zeros((10, 10, n_actions))  # speed_bin x error_bin x actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_error = 0.0

    def _discretize(self, speed, error):
        speed_bin = min(int(speed * 2), 9)
        error_bin = min(int(abs(error) / 2), 9)
        return speed_bin, error_bin

    def choose_action(self, speed, error):
        s, e = self._discretize(speed, error)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[s][e])

    def learn(self, speed, error, action, reward, next_error):
        s, e = self._discretize(speed, error)
        s_next, e_next = self._discretize(speed, next_error)
        best_next_q = np.max(self.q_table[s_next][e_next])
        current_q = self.q_table[s][e][action]
        self.q_table[s][e][action] += self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.last_error = next_error
