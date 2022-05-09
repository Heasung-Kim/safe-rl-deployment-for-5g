"""
    This code is based on the discrete BCQ. Please see the following link,
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
    """

    If the constrained batch's samples do not follow an uniform distribution in terms of action selection,
    discrete BCQ Logic can be applied.
    Note that this function will not be working for the uniform action distribution, and it does not affect the performance.
    """
    def __init__(self, state_shape, action_shape):
        super(Action_Probability, self).__init__()
        self.layer1 = nn.Linear(state_shape, General_Depth)
        self.layer2 = nn.Linear(General_Depth, General_Depth)
        self.layer3 = nn.Linear(General_Depth, action_shape)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1)


class BCMQ(object):
    def __init__(self, config, action_shape, state_shape, device):
        self.state_shape = (-1, state_shape)
        self.action_shape = action_shape
        self.device = device
        self.config = config
        self.learning_rate = config["algorithm_config"]["learning_rate"]
        self.threshold = config["algorithm_config"]["bcq_threshold"]
        self.discount = config["algorithm_config"]["discount_factor"]

        # Neural Networks
        self.Q = Action_Value(state_shape, action_shape).to(self.device)
        self.G = Action_Probability(state_shape, action_shape).to(self.device)

        # Target Evaluation Technique (DQN)
        self.Q_target = copy.deepcopy(self.Q)

        # Two optimizers for action-value update and approximated policy (for constrained batch)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate)

        # Do not change these things
        self._tau = 0.01

    def select_model_based_rollout_action(self, state, env):
        """

        Generate Top 2 action candidates and do 1-step tree search based on the actions.
        Given next expected state, calculate action-value function one more.
        Pick the best action which generates the maximum action-value function at the next time slot.

        :param state: state from gym env class
        :param env: Environment instance having "expect_next_state" function.
        :return: 1-dim discrete action
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q = self.Q(state)
            log_action_probability = self.G(state)
            action_probability = log_action_probability.exp()

            # ===========================================================================================================
            # BCQ Logic: code from "Use large negative number to mask actions from argmax from https://github.com/sfujim/BCQ/ "
            action_probability = (action_probability / action_probability.max(1, keepdim=True)[0] > self.threshold).float()
            candidates =  (action_probability * q + (1. - action_probability) * -1e8)
            values,indices = candidates.topk(2)
            # ===========================================================================================================

            # For the state dynamics' deterministic property, we can get the "estimated" next state
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

        # Get action value, it should be size "action dim"
        q_s = self.Q(next_state)
        log_action_probability = self.G(next_state)

        # ===========================================================================================================
        # BCQ Logic: code from "Use large negative number to mask actions from argmax from https://github.com/sfujim/BCQ/ "
        next_action_probability = log_action_probability.exp()
        next_action_probability = (next_action_probability / next_action_probability.max(1, keepdim=True)[0] > self.threshold).float()
        next_action = (next_action_probability * q_s+ (1 - next_action_probability) * -1e8).argmax(1, keepdim=True)
        # ===========================================================================================================
        q_s= self.Q_target(next_state)

        # if done flag is true, then the epi (RF) is aborted, no more discounted reward sum.
        label_action_value = reward + (1-done) * self.discount * q_s.gather(1, next_action).reshape(-1, 1)

        prediction_action_value = self.Q(state).gather(1, action)
        action_probability = self.G(state)

        # Parameters are updated in direction of minimizing mean-squared error.
        q_loss = F.mse_loss(prediction_action_value, label_action_value)
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
        for q_params, target_q_params in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_q_params.data.copy_(self._tau * q_params.data + (1 - self._tau) * target_q_params.data)

