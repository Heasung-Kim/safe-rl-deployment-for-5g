"""

    Part of this code are based on the BCQ thresholding logic. Please see the following link,
    https://github.com/sfujim/BCQ/
    Fujimoto, Scott, David Meger, and Doina Precup. "Off-policy deep reinforcement learning without exploration." International Conference on Machine Learning. PMLR, 2019.
    Fujimoto, Scott, et al. "Benchmarking batch deep reinforcement learning algorithms." arXiv preprint arXiv:1910.01708 (2019).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
General_Depth = 24


class Action_Value(nn.Module):
    """
    Action Value function, Q(s,a)
    Call instance makes action-shape dimension vector.
    """
    def __init__(self, state_shape, action_shape):
        super(Action_Value, self).__init__()
        self.layer1 = nn.Linear(state_shape, General_Depth)
        self.layer2 = nn.Linear(General_Depth, action_shape)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.layer2(x)
        return x


class Action_Probability(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Action_Probability, self).__init__()
        self.layer1 = nn.Linear(state_shape, General_Depth)
        self.layer2 = nn.Linear(General_Depth, General_Depth)
        self.layer3 = nn.Linear(General_Depth, action_shape)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1), x


class BCMQ(object):
    def __init__(self, config, action_shape, state_shape, device):
        self.state_shape = (-1, state_shape)
        self.action_shape = action_shape
        self.device = device
        self.config = config
        self.learning_rate = config["algorithm_config"]["learning_rate"]
        self.threshold = config["algorithm_config"]["bcq_threshold"]
        self.discount = config["algorithm_config"]["discount_factor"]
        self.Q = Action_Value(state_shape, action_shape).to(self.device)
        self.G = Action_Probability(state_shape, action_shape).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate)

        # Do not change these things
        self._tau = 0.01

    def select_action(self, state):
        """
        BCQ action selection logic from https://github.com/sfujim/BCQ/
        Fujimoto, Scott, David Meger, and Doina Precup. "Off-policy deep reinforcement learning without exploration." International Conference on Machine Learning. PMLR, 2019.
        Fujimoto, Scott, et al. "Benchmarking batch deep reinforcement learning algorithms." arXiv preprint arXiv:1910.01708 (2019).

        :param state:
        :return:
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q = self.Q(state)
            log_action_probability, i = self.G(state)
            action_probability = log_action_probability.exp()
            action_probability = (action_probability / action_probability.max(1, keepdim=True)[0] > self.threshold).float()
            return int((action_probability * q + (1. - action_probability) * -1e8).argmax(1))

    def select_model_based_rollout_action(self, state, env):
        """

        Generate Top 2 action candidates and do 1-step tree search based on the actions.
        Given next expected state, calculate action-value function one more.
        Pick the best action which generates the maximum action-value function at the next time slot.

        :param state:
        :param env: Environment instance having "expect_next_state" function.
        :return:
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q = self.Q(state)
            log_action_probability, _ = self.G(state)
            action_probability = log_action_probability.exp()

            # Use large negative number to mask actions from argmax (borrowed from https://github.com/sfujim/BCQ/)
            action_probability = (action_probability / action_probability.max(1, keepdim=True)[0] > self.threshold).float()
            candidates =  (action_probability * q + (1. - action_probability) * -1e8)
            values,indices = candidates.topk(2)

            next_state_1 = env.expect_next_state(state, int(indices[0][0]))
            next_state_2 = env.expect_next_state(state, int(indices[0][1]))

            next_state_1 = torch.FloatTensor(next_state_1).reshape(self.state_shape).to(self.device)
            ext_timeslot_Q1 = self.Q(next_state_1)

            next_state_2 = torch.FloatTensor(next_state_2).reshape(self.state_shape).to(self.device)
            ext_timeslot_Q2 = self.Q(next_state_2)

            if np.max(ext_timeslot_Q1.numpy()) > np.max(ext_timeslot_Q2.numpy()):
                action = int(indices[0][0])
            else:
                action = int(indices[0][1])
            return action


    def train(self, replay_buffer):
        state, action, next_state, reward, done = replay_buffer.get_sample()

        # Compute the target Q value
        with torch.no_grad():
            q = self.Q(state)
            log_action_probability, i = self.G(state)
            action_probability = log_action_probability.exp()
            action_probability = (action_probability / action_probability.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax (borrowed from https://github.com/sfujim/BCQ/)
            next_action = (action_probability * q + (1 - action_probability) * -1e8).argmax(1, keepdim=True)

            q = self.Q_target(next_state)

            # if done flag is true, then the epi (RF) is aborted, no more discounted reward sum.
            target_Q = reward + (1-done) * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)
        action_probability, i = self.G(state)

        # Parameters are updated in direction of minimizing mean-squared error.
        q_loss = F.mse_loss(current_Q, target_Q)
        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        # Negative log likelihood loss
        # Our selected action should be followed by the action probability.
        g_loss = F.nll_loss(action_probability, action.reshape(-1))
        self.G_optimizer.zero_grad()
        g_loss.backward()
        self.G_optimizer.step()

        self.soft_target_update()

    def soft_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

