import random

import numpy as np

FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0


class QLearningAgent:
    def __init__(self, lr, gamma, track_length, epsilon=0, policy='greedy'):
        """
        A function for initializing your agent
        :param lr: learning rate
        :param gamma: discount factor
        :param track_length: how far the ends of the track are from the origin.
            e.g., while track_length is 2.4,
            the x-coordinate of the left end of the track is -2.4,
            the x-coordinate of the right end of the track is 2.4,
            and x-coordinate of the the cart is 0 initially.
        :param epsilon: epsilon for the mixed policy
        :param policy: can be 'greedy' or 'mixed'
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.track_length = track_length
        self.policy = policy
        random.seed(11)
        np.random.seed(11)
        self.q_table = {}
        self.freq_table = {}
        self.max_ang_vel = .01
        self.max_vel = .01
        pass

    def reset(self):
        """
        you may add code here to re-initialize your agent before each trial
        :return:
        """

        # for i in range (256):

        pass

    def get_action(self, x, x_dot, theta, theta_dot):
        """
        main.py calls this method to get an action from your agent
        :param x: the position of the cart
        :param x_dot: the velocity of the cart
        :param theta: the angle between the cart and the pole
        :param theta_dot: the angular velocity of the pole
        :return:
        """
        if self.policy == 'mixed' and random.random() < self.epsilon:
            action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        else:
            # fill your code here to get an action from your agent
            discretized_state = self.discretize((x, x_dot, theta, theta_dot))
            if discretized_state not in self.q_table:
                action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
            else:
                if self.q_table[discretized_state][FORWARD_ACCEL] > self.q_table[discretized_state][BACKWARD_ACCEL]:
                    action = FORWARD_ACCEL
                else:
                    if self.q_table[discretized_state][BACKWARD_ACCEL] > self.q_table[discretized_state][FORWARD_ACCEL]:
                        action = BACKWARD_ACCEL
                    else:
                        action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
                # print(self.q_table[discretized_state][action])
        return action

    def update_q(self, prev_state, prev_action, cur_state, reward):

        """
        main.py calls this method so that you can update your Q-table
        :param prev_state: previous state, a tuple of (x, x_dot, theta, theta_dot)
        :param prev_action: previous action, FORWARD_ACCEL or BACKWARD_ACCEL
        :param cur_state: current state, a tuple of (x, x_dot, theta, theta_dot)
        :param reward: reward, 0.0 or -1.0
        e.g., if we have S_i ---(action a, reward)---> S_j, then
            prev_state is S_i,
            prev_action is a,
            cur_state is S_j,
            rewards is reward.
        :return:
        """
        if abs(prev_state[1]) > self.max_vel:
            self.max_vel = abs(prev_state[1])
        if abs(prev_state[3]) > self.max_ang_vel:
            self.max_ang_vel = abs(prev_state[3])
        prev_discretized_state = self.discretize(prev_state)
        cur_discretized_state = self.discretize(cur_state)

        if prev_discretized_state not in self.freq_table.keys():
            self.freq_table[prev_discretized_state] = {}
            self.freq_table[prev_discretized_state][FORWARD_ACCEL] = 0
            self.freq_table[prev_discretized_state][BACKWARD_ACCEL] = 0

        if prev_discretized_state not in self.q_table.keys():
            self.q_table[prev_discretized_state] = {}
            self.q_table[prev_discretized_state][FORWARD_ACCEL] = 0
            self.q_table[prev_discretized_state][BACKWARD_ACCEL] = 0

        if cur_discretized_state not in self.freq_table.keys():
            self.freq_table[cur_discretized_state] = {}
            self.freq_table[cur_discretized_state][FORWARD_ACCEL] = 0
            self.freq_table[cur_discretized_state][BACKWARD_ACCEL] = 0

        if cur_discretized_state not in self.q_table.keys():
            self.q_table[cur_discretized_state] = {}
            self.q_table[cur_discretized_state][FORWARD_ACCEL] = 0
            self.q_table[cur_discretized_state][BACKWARD_ACCEL] = 0
        # print(self.q_table[cur_discretized_state][FORWARD_ACCEL], '      ', self.q_table[cur_discretized_state][BACKWARD_ACCEL])

        prev_q = self.q_table[prev_discretized_state][prev_action]
        lr_times_freq = self.lr * 1  # self.freq_table[prev_discretized_state][prev_action]
        third_expr = reward + max(self.q_table[cur_discretized_state][0],
                                  self.q_table[cur_discretized_state][1]) - prev_q
        new_q = prev_q + lr_times_freq * third_expr
        self.q_table[prev_discretized_state][prev_action] = new_q

    def discretize(self, state):
        x_prev = round(state[0] * (7 / self.track_length))
        prev_vel = round(state[1] * 2.7)
        prev_angle = round(state[2] * 25)
        prev_ang_v = round(state[3] * 2.7)
        return x_prev, prev_vel, prev_angle, prev_ang_v

    # you may add more methods here for your needs. E.g., methods for discretizing the variables.


if __name__ == '__main__':
    pass
