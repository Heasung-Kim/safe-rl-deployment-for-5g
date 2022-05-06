import gym
from gym import spaces
import numpy as np
import random
import os
import pandas as pd
import pickle
import itertools
from global_config import ROOT_DIR
import scipy.io as sio
import os
import pickle

import yaml
import os

from env.data_generators.scenario_02 import Scenario


class UserEquipment(object):
    """ Custom Environment that follows gym interface """

    def __init__(self, config, ue_unique_num):

        self.config = config  # env = Environment(config, train_mode=True)
        self.data_storage_path = os.path.join(ROOT_DIR, "data", self.config["config_name"])
        self.unique_num = ue_unique_num

        self._connected_bs = -1 # -1 means that this ue is not connected to any of BS (0~6)
        self.data = None
        self.x_coordinates = None
        self.y_coordinates = None
        self.channel_matrix = None

        self.time_index = 0

        self.scenario = Scenario(config=self.config)


    def result_analysis(self, epi_data):
        self.epi_data = epi_data
        self.ue_connection_status = int(epi_data[self.time_index]["ue_connection_status"][self.unique_num])
        self.ue_serving_status = int(epi_data[self.time_index]["ue_serving_status"][self.unique_num])

        self.current_channel_for_all_bs = self.channel_matrix[:, self.time_index, :]


    def set_connected_bs(self, bs_num):
        self._connected_bs = bs_num

    def get_connected_bs(self):
        return self._connected_bs

    def generate_immediate_scenario_data(self):
        self.data = self.scenario.generate_immediate_single_ue_data(self.unique_num)

        self.x_coordinates = self.data["ue_coordinates"][0]
        self.y_coordinates = self.data["ue_coordinates"][1]
        self.channel_matrix = self.data["channel_matrix"]

        self.x = self.x_coordinates[self.time_index]
        self.y = self.y_coordinates[self.time_index]

    def set_scenario_data(self, epi_idx):
        with open(os.path.join(self.data_storage_path, "epi_"+str(epi_idx), "ue_" + str(self.unique_num), 'data.pickle'), 'rb') as handle:
            data = pickle.load(handle)
        self.data = data

        self.x_coordinates = self.data["ue_coordinates"][0]
        self.y_coordinates = self.data["ue_coordinates"][1]
        self.channel_matrix = self.data["channel_matrix"]

        self.x = self.x_coordinates[self.time_index]
        self.y = self.y_coordinates[self.time_index]


    def set_scenario_mat_data(self, epi_idx, MISO=True):
        self.data_storage_path = os.path.join(ROOT_DIR, "data", self.config["config_name"])
        data = sio.loadmat(os.path.join(self.data_storage_path, "epi_"+str(epi_idx+1), "ue_" + str(self.unique_num), 'data.mat'))
        self.data = data

        self.channel_matrix = self.data["channel_state"]
        if MISO==True:
            self.channel_matrix = self.channel_matrix[:,:,:,0]

        self.__ue_position_estimate(self.data["ue_position"], self.data["ue_velocity"], epi_length=self.channel_matrix.shape[1])

        self.x = self.x_coordinates[self.time_index]
        self.y = self.y_coordinates[self.time_index]

    def __ue_position_estimate(self, initial_point, velocity, epi_length, time_slot_length=0.04):
        self.x_coordinates=[]
        self.y_coordinates=[]
        for i in range(epi_length):

            self.x_coordinates.append(initial_point[0][0] + velocity[0][0] * 0.04 * i)
            self.y_coordinates.append(initial_point[0][1] + velocity[0][1] * 0.04 * i)

        self.x_coordinates = np.array(self.x_coordinates)
        self.y_coordinates = np.array(self.y_coordinates)

    def move(self):
        self.time_index = self.time_index + 1
        self.x = self.x_coordinates[self.time_index]
        self.y = self.y_coordinates[self.time_index]

        self.current_channel_for_all_bs = self.channel_matrix[:, self.time_index, :]
        self.ue_connection_status = int(self.epi_data[self.time_index]["ue_connection_status"][self.unique_num])
        self.ue_serving_status = int(self.epi_data[self.time_index]["ue_serving_status"][self.unique_num])
        self.current_channel_for_all_bs = self.channel_matrix[:,self.time_index,:]


if __name__ == '__main__':
    # get configuration
    with open(os.path.join(ROOT_DIR,"config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    myue = UserEquipment(config, 1)
    myue.set_scenario_mat_data(1)


