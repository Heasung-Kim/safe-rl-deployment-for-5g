import numpy as np
from numpy.linalg import inv, norm
import time


class BaseStation(object):
    def __init__(self, config, base_station_unique_num, ues):
        """
        :param train_mode: True = train mode, False = test mode
        """

        self.config = config
        self.unique_number = base_station_unique_num

        self.connected_users = []
        self.ues = ues

        self.x_coord = config["system_model_config"]["bs_x_coordinate"][self.unique_number]
        self.y_coord = config["system_model_config"]["bs_y_coordinate"][self.unique_number]
        self.coverage = config["system_model_config"]["cell_radius"]
        self.num_antenna = config["system_model_config"]["M_ULA"]
        self.backhaul_delay = config["system_model_config"]["backhaul_delay"]

        self.inner_timer = 0
        self.global_timer = 0

        self.served_ue = []

        # Wireless Settings
        self.noise_power = 1.38e-23 * 290 * 15000

        self.transmit_power = 10



    def set_global_timer(self, time):
        self.global_timer = time

    def force_connect_ue(self, ue_num):
        self.connected_users = [ue_num]
        self.served_ue = [ue_num]

    def connect_to_ue(self, ue_num):
        self.connected_users.append(ue_num)

    def disconnect_to_ue(self, ue_num):
        self.connected_users.remove(ue_num)

    def select_served_ues(self):
        if len(self.connected_users) > 0:
            self.inner_timer = (self.inner_timer + 1) % len(self.connected_users)
            self.served_ue = [self.connected_users[self.inner_timer]]
            return [self.connected_users[self.inner_timer]]
        else:
            self.served_ue = []
            return []

