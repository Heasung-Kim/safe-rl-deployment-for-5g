from global_config import ROOT_DIR, DATA_STORAGE
import numpy as np
import torch
import os
from pathlib import Path

class Replay_Buffer(object):
    """

    """
    def __init__(self, config, state_shape, action_shape, directory):
        self.config = config
        self.max_num_sample = config["algorithm_config"]["buffer_max_size"]
        self.state = np.zeros( (self.max_num_sample,) + state_shape)
        self.action = np.zeros( (self.max_num_sample,) + action_shape + (1,))
        self.next_state = np.zeros( (self.max_num_sample,) + state_shape)
        self.reward =np.zeros( (self.max_num_sample,1))
        self.done = np.zeros( (self.max_num_sample,1))

        self.buffer_pointer = 0
        self.save_directory = directory
        self.batch_size = self.config["algorithm_config"]["batch_size"]

    def add_sample(self, state, action, next_state, reward, done):
        self.state[int(self.buffer_pointer % self.max_num_sample)] = state
        self.action[int(self.buffer_pointer % self.max_num_sample)] =  action
        self.next_state[int(self.buffer_pointer % self.max_num_sample)] = next_state
        self.reward[int(self.buffer_pointer % self.max_num_sample)] = reward
        self.done[int(self.buffer_pointer % self.max_num_sample)] = done
        self.buffer_pointer += 1

    def get_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        ind = np.random.randint(0, self.max_num_sample, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.LongTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.done[ind])
        )

    def save(self):
        Path(self.save_directory).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(self.save_directory, "state.npy"), self.state)
        np.save(os.path.join(self.save_directory, "action.npy"), self.action)
        np.save(os.path.join(self.save_directory, "next_state.npy"), self.next_state)
        np.save(os.path.join(self.save_directory, "reward.npy"), self.reward)
        np.save(os.path.join(self.save_directory, "done.npy"), self.done)

    def load(self):
        np.load(os.path.join(self.save_directory, "state.npy"))
        np.load(os.path.join(self.save_directory, "action.npy"))
        np.load(os.path.join(self.save_directory, "next_state.npy"))
        np.load(os.path.join(self.save_directory, "reward.npy"))
        np.load(os.path.join(self.save_directory, "done.npy"))
