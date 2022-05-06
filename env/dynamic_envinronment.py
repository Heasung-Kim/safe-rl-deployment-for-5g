import gym
from gym import spaces
import numpy as np
import random
import os
import pandas as pd
import pickle
import math
import itertools
from global_config import ROOT_DIR, DATA_STORAGE
from env.base_station import BaseStation
from env.user_equipment import UserEquipment
from numpy.linalg import inv, norm
from env.channel.channel import compute_bf_vector
from collections import deque
from pathlib import Path
MAXIMUM_FLOAT = np.inf

class Environment(gym.Env):
    """ Custom Environment that follows gym interface """

    def __init__(self, config, TRAIN_MODE):
        """
        :param train_mode: True = train mode, False = test mode
        """
        super(Environment, self).__init__()

        self.TRAIN_MODE = TRAIN_MODE
        if self.TRAIN_MODE is False:
            self.seed(0)
        self.config = config  # env = Environment(config, train_mode=True)
        self.t = 0

        # Wireless Settings
        self.noise_power = self.config["system_model_config"]["noise_power"]
        self.MAX_SINR = 70 # Greatest possible SINR
        self.max_transmission_power = 40


        self._max_episode_steps = config["episode_config"]["episode_length"]
        self.train_epi_indices = config["episode_config"]["train_epi_indices"]
        self.train_epi_index = 0
        self.test_epi_indices = config["episode_config"]["test_epi_indices"]
        self.test_epi_index = 0

        # build system model
        self.user_equipments = [UserEquipment(self.config, ue_unique_num=i) for i in
                           range(self.config["system_model_config"]["num_ue"])]
        self.num_ue = len(self.user_equipments)

        self.base_stations = [BaseStation(self.config, base_station_unique_num=i, ues=self.user_equipments) for i in
                         range(self.config["system_model_config"]["num_bs"])]
        self.num_bs = len(self.base_stations)

        # Internal Elements
        self._ue_connection_status = np.zeros(shape=(self.num_ue), dtype=int)
        self._ue_serving_status = np.zeros(shape=(self.num_ue), dtype=int)
        self._ue_codebook_indices = np.zeros(shape=(self.num_ue), dtype=int)
        self._ue_transmission_power = 20.0 *np.ones(shape=(self.num_ue), dtype=int) # Watt
        self.SINR_queue_size = 10
        self._ue_SINR_queues = [deque(maxlen=self.SINR_queue_size) for _ in range(self.num_ue)]

        self.transmission_power_queue_size = 10
        self._ue_trasmission_power_queues = [deque(maxlen=self.transmission_power_queue_size) for _ in range(self.num_ue)]
        for u_i in range(self.num_ue):
            queue = self._ue_trasmission_power_queues[u_i]
            for i in range(self.transmission_power_queue_size):
                queue.append(self._ue_transmission_power[u_i])

        self._ue_SINRs = 20*np.ones(shape=(self.num_ue), dtype=float)
        for u_i in range(self.num_ue):
            queue = self._ue_SINR_queues[u_i]
            for i in range(self.SINR_queue_size):
                queue.append(self._ue_SINRs[u_i])

        self.internal_reward_history = []
        # =====================================================================================================================
        # The method of defining member variables below (center frequency to reward min-max) follows the code method of Dr faris.
        # https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks
        # =====================================================================================================================
        self.center_frequency = self.config["system_model_config"]["center_frequency"]
        self.M_ULA = self.config["system_model_config"]["M_ULA"]
        self.min_sinr = -3 # in dB
        self.sinr_target = 5 + 10*np.log10(self.M_ULA) # in dB.
        self.num_beamforming_codes = self.M_ULA

        # for Beamforming
        self.use_beamforming = True
        self.Np = 4  # from 3 to 5 for mmWave

        self.num_precoders = self.M_ULA
        self.precoding_codebook = np.zeros([self.M_ULA, self.num_precoders], dtype=complex)
        self.theta_n = math.pi * np.arange(start=0., stop=1., step=1. / (self.M_ULA))
        # Beamforming codebook F
        for codebook_idx in np.arange(self.num_precoders):
            f_n = compute_bf_vector(theta=self.theta_n[codebook_idx],  f_c=self.center_frequency, M_ULA=self.M_ULA)
            self.precoding_codebook[:,codebook_idx] = f_n

        self.reward_min = -20
        self.reward_max = 100
        # =====================================================================================================================
        # The above of defining member variables below (center frequency to reward min-max) follows the code method of Dr faris.
        # https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks
        # =====================================================================================================================


        # Reinforcement Learning Setting
        self.action_space = spaces.Discrete(int(2**(self.num_ue + self.num_ue)))  # UE powers, UE Codebook Indices
        self.state_space_lower_bound = []
        self.state_space_upper_bound = []
        for u_i in range(self.num_ue):
            self.state_space_lower_bound.append(0)
            self.state_space_lower_bound.append(0)
            self.state_space_upper_bound.append(self.config["system_model_config"]["inter_site_distance"])
            self.state_space_upper_bound.append(self.config["system_model_config"]["inter_site_distance"])


        for u_i in range(self.num_ue):
            self.state_space_lower_bound.append(0)
            self.state_space_upper_bound.append(self.max_transmission_power)


        for u_i in range(self.num_ue):
            self.state_space_lower_bound.append(0)
            self.state_space_upper_bound.append(self.num_beamforming_codes-1)

        """
        for u_i in range(self.num_ue * self.SINR_queue_size):
            self.state_space_lower_bound.append(-self.MAX_SINR)
            self.state_space_upper_bound.append(self.MAX_SINR)
        """

        self.observation_space = spaces.Box(low=np.array(self.state_space_lower_bound),
                                            high=np.array(self.state_space_upper_bound),
                                            dtype=np.float32)

    def _connect_bs_ue(self, timeslot):

        # store connection history
        for ue_idx in range(self.num_ue):
            ue = self.user_equipments[ue_idx]
            ue.set_connected_bs(ue_idx)


        for bs_idx in range(self.num_bs):
            bs = self.base_stations[bs_idx]
            bs.force_connect_ue(bs_idx)

        return

    def _connect_bs_ue_scheduling(self, timeslot):
        for ue_idx in range(self.num_ue):
            ue = self.user_equipments[ue_idx]
            #print(ue_idx)
            ue_x_coord = ue.x_coordinates[self.t]
            ue_y_coord = ue.y_coordinates[self.t]
            bs_candidate = -1

            if ue.get_connected_bs() == -1: # ue is not connected to the BSs that we are interested in...
                bs_candidate = -1
                min_dist = 1e10
                for bs_idx in range(self.num_bs):
                    bs = self.base_stations[bs_idx]
                    bs_x_coord, bs_y_coord = bs.x_coord, bs.y_coord
                    ue_bs_distance = math.sqrt( (ue_x_coord - bs_x_coord) ** 2 + (ue_y_coord - bs_y_coord) ** 2)
                    if ue_bs_distance < bs.coverage and min_dist > ue_bs_distance:

                        ue_cm = ue.channel_matrix[bs_idx][self.t]
                        if norm(ue_cm) < 1e-15:
                            continue
                        bs_candidate = bs_idx
                        min_dist = ue_bs_distance

                # UE - BS Link
                ue.set_connected_bs(bs_candidate)
                if bs_candidate >= 0:
                    bs = self.base_stations[bs_candidate]
                    bs.connect_to_ue(ue_idx)

            else: # case that ue is currently connected to some bs
                bs = self.base_stations[ue.get_connected_bs()]
                cbs_x_coord, cbs_y_coord = bs.x_coord, bs.y_coord
                ue_bs_distance = math.sqrt((ue_x_coord - cbs_x_coord) ** 2 + (ue_y_coord - cbs_y_coord) ** 2)
                if ue_bs_distance > bs.coverage :
                    # The previous connected bs is not available.
                    bs.disconnect_to_ue(ue_idx)
                    # if ue is outside of the previous bs coverage, new bs is allocated.
                    bs_candidate = -1
                    min_dist = MAXIMUM_FLOAT
                    for bs_idx in range(self.num_bs):
                        bs = self.base_stations[bs_idx]
                        bs_x_coord, bs_y_coord = bs.x_coord, bs.y_coord
                        ue_bs_distance = math.sqrt((ue_x_coord - bs_x_coord) ** 2 + (ue_y_coord - bs_y_coord) ** 2)
                        if ue_bs_distance < bs.coverage and min_dist > ue_bs_distance:
                            ue_cm = ue.channel_matrix[bs_idx][self.t]
                            if norm(ue_cm) < 1e-15:
                                continue
                            bs_candidate = bs_idx
                            min_dist = ue_bs_distance

                    # UE - BS Link
                    ue.set_connected_bs(bs_candidate)
                    if bs_candidate >= 0:
                        bs = self.base_stations[bs_candidate]
                        bs.connect_to_ue(ue_idx)
                else:
                    pass

        # store connection history
        for ue_idx in range(self.num_ue):
            ue = self.user_equipments[ue_idx]
            self.bs_ue_connection_matrix[timeslot][ue_idx] = ue.get_connected_bs()
        return


    def _get_current_state(self):
        """

        :return:
        """
        state = []

        for u_i in range(self.num_ue):
            x = self.user_equipments[u_i].x_coordinates[self.t]
            y = self.user_equipments[u_i].x_coordinates[self.t]

            x = (x-self.state_space_lower_bound[0]) / (self.state_space_upper_bound[0] - self.state_space_lower_bound[0])
            y = (y-self.state_space_lower_bound[0]) / (self.state_space_upper_bound[0] - self.state_space_lower_bound[0])

            state.append(x)
            state.append(y)


        for u_i in range(self.num_ue):
            transmission_power = self._ue_transmission_power[u_i]
            state.append(transmission_power)


        for u_i in range(self.num_ue):
            codebook_index = self._ue_codebook_indices[u_i]  #/ self.num_beamforming_codes # currently, we don't use normalization
            state.append(codebook_index)

        """
        # For baseic setting, SINR will not be included in the state tuple
        for u_i in range(self.num_ue):
            queue = list(self._ue_SINR_queues[u_i])
            state.extend(np.array(queue)/ 100.0)
        """

        return np.array(state)

    def _calculate_UE_SINR(self, user_index):

        channel_matrix = self.user_equipments[user_index].channel_matrix
        serving_BS_index = self.user_equipments[user_index].get_connected_bs()
        serving_channel = channel_matrix[serving_BS_index][self.t]

        serving_transmission_power = self._ue_transmission_power[user_index]


        serving_power = serving_transmission_power * abs(np.dot(serving_channel.conj(),
                                                                 self.precoding_codebook[:, self._ue_codebook_indices[user_index]])) ** 2

        total_interference_power = 0

        for interfering_u_i in range(self.num_ue-1):
            if interfering_u_i == user_index:
                continue
            else:
                interfering_BS_index = self.user_equipments[interfering_u_i].get_connected_bs()
                interfering_channel = channel_matrix[interfering_BS_index][self.t]
                interfering_transmission_power = self._ue_transmission_power[interfering_u_i]
                interference_power = interfering_transmission_power * abs(np.dot(interfering_channel.conj(),
                                                                        self.precoding_codebook[:,
                                                                        self._ue_codebook_indices[interfering_u_i]])) ** 2

                total_interference_power += interference_power

        total_interference_power += self.noise_power
        db_scale_SINR = 10 * np.log10(serving_power / total_interference_power)

        return serving_power, total_interference_power, db_scale_SINR

    def reset(self):
        if self.TRAIN_MODE:
            try:
                epi_idx = self.train_epi_indices[self.train_epi_index % len(self.train_epi_indices)]
                self.train_epi_index += 1
            except:
                raise NotImplementedError
        else:
            epi_idx = self.test_epi_indices[self.test_epi_index]
            #self.test_epi_index += 1

        for ue_idx in range(self.num_ue):
            #self.user_equipments[ue_idx].set_scenario_data(epi_idx=epi_idx) # Data version
            self.user_equipments[ue_idx].generate_immediate_scenario_data() # Immediate version
            #self.user_equipments[ue_idx].set_scenario_mat_data(epi_idx=epi_idx)


        for ue_idx in range(self.num_ue):
            self._ue_codebook_indices[ue_idx] = np.random.randint(low=0,high=self.M_ULA)
            self._ue_transmission_power[ue_idx] = np.random.uniform(low=1, high=self.max_transmission_power/2)

        self.t = 0
        #self.bs_ue_connection_matrix = -1 * np.ones(shape=(self._max_episode_steps, self.num_ue))
        self.bs_ue_connection_matrix = np.array([np.arange(self.num_bs) for _ in range(self._max_episode_steps)])
        self._connect_bs_ue(timeslot=self.t)

        return self._get_current_state()


    def step(self, action):

        # System Dynamics
        self._connect_bs_ue(timeslot=self.t)

        # select served ue
        served_ues = []
        for bs_idx in range(self.num_bs):
            self.base_stations[bs_idx].set_global_timer(self.t)
            served_ues.extend(self.base_stations[bs_idx].select_served_ues())


        # Apply Actions
        binary_action = format(action, '0'+str(int(np.log2(self.action_space.n)))+'b')
        action_searching_idx = 0

        # Power Control
        for u_i in range(self.num_ue):
            if binary_action[action_searching_idx] == '0':
                self._ue_transmission_power[u_i] = np.clip([10 ** (1/10.) * self._ue_transmission_power[u_i]], 0, self.max_transmission_power)
            elif binary_action[action_searching_idx] == '1':
                self._ue_transmission_power[u_i] = np.clip([10 ** (-1/10.) * self._ue_transmission_power[u_i]], 0 , self.max_transmission_power)
            else:
                raise NotImplementedError
            action_searching_idx += 1

        # Beamforming Codebook Indices Control
        for u_i in range(self.num_ue):
            if binary_action[action_searching_idx] == '0':
                self._ue_codebook_indices[u_i] = np.clip([self._ue_codebook_indices[u_i]-1], 0, self.num_precoders-1)[0]
            elif binary_action[action_searching_idx] == '1':
                self._ue_codebook_indices[u_i] = np.clip([self._ue_codebook_indices[u_i]+1], 0, self.num_precoders-1)[0]
            else:
                raise NotImplementedError
            action_searching_idx += 1

        # Calculate SINR
        for u_i in range(self.num_ue):
            serving_power, total_interference_power, db_scale_SINR = self._calculate_UE_SINR(user_index = u_i)
            self._ue_SINRs[u_i] = db_scale_SINR

            queue = self._ue_SINR_queues[u_i]
            if db_scale_SINR == -np.inf:
                break
            queue.append(db_scale_SINR)

        # Store TP
        for u_i in range(self.num_ue):
            queue = self._ue_trasmission_power_queues[u_i]
            queue.append(self._ue_transmission_power[u_i])

        # Reward calculation phase
        """ Transmission Power Minimization Problem """
        """
        reward = 0
        done = False
        for u_i in range(self.num_ue):
            queue = self._ue_SINR_queues[u_i]
            if np.average(queue) > -20:
                reward += (60 - self._ue_transmission_power[u_i]) / 10
            else:

                done = True
            #reward = np.sum(self._ue_SINRs) / 100.0
        if done == True:
            reward += - 100
        #
        """



        # 5G Network Setting SNR Maximization Problem
        # Abortion condition and reach_done condition is given by
        # Mismar, Faris B., Brian L. Evans, and Ahmed Alkhateeb.
        # "Deep reinforcement learning for 5G networks: Joint beamforming, power control, and interference coordination."
        # IEEE Transactions on Communications 68.3 (2019): 1581-1592.

        reach_done = (self._ue_transmission_power[0] <= self.max_transmission_power) \
               and (self._ue_transmission_power[0] >= 0) \
               and (self._ue_transmission_power[1] <= self.max_transmission_power) \
               and (self._ue_transmission_power[1] >= 0) \
               and (self._ue_SINRs[0] >= self.min_sinr) \
               and (self._ue_SINRs[1] >= self.min_sinr) \
               and (self._ue_SINRs[0] >= self.sinr_target) \
               and (self._ue_SINRs[1] >= self.sinr_target)

        abort = (self._ue_transmission_power[0] > self.max_transmission_power) \
                or (self._ue_transmission_power[1] > self.max_transmission_power) \
                or (self._ue_SINRs[0] < self.min_sinr) \
                or (self._ue_SINRs[1] < self.min_sinr) \
                or (self._ue_SINRs[0] > 70) \
                or (self._ue_SINRs[1] > 70)  # or (received_sinr < 10) or (received_ue2_sinr < 10)
        reward = self._ue_SINRs[0] + self._ue_SINRs[1]

        if abort == True:
            reach_done = False
            reward = self.reward_min
        elif reach_done:
            reward += self.reward_max

        for user_idx in range(self.num_ue):
            self._ue_connection_status[user_idx] = self.user_equipments[user_idx].get_connected_bs()
            self._ue_serving_status[user_idx] = 0
            if self._ue_connection_status[user_idx] != -1:
                bs = self.base_stations[self.user_equipments[user_idx].get_connected_bs()]
                if user_idx in bs.served_ue:
                    self._ue_serving_status[user_idx] = 1
                else:
                    self._ue_serving_status[user_idx] = 0

        next_state = self._get_current_state()

        self.t = self.t + 1
        info = {"reach_done": reach_done,
                "abort": abort,
                "all_user_rate": self._ue_SINRs,
                "ue_connection_status": self._ue_connection_status,
                "ue_serving_status": self._ue_serving_status}
        #print(self._ue_serving_status, "sum:", np.sum(self._ue_serving_status))
        done = self.t==self.config["episode_config"]["episode_length"]  or abort == True

        self.reach_done = reach_done
        self.abort = abort
        self.internal_reward_history.append(reward)
        return next_state, reward, done, info


    def expect_next_state(self, state, action):
        try:
            state = state.numpy()
        except:
            pass
        state = state[0]
        _ue_transmission_power = state[4:6]
        _ue_codebook_indices = state[6:8]

        # Apply Actions
        binary_action = format(action, '0'+str(int(np.log2(self.action_space.n)))+'b')
        action_searching_idx = 0

        # Power Control
        for u_i in range(self.num_ue):
            if binary_action[action_searching_idx] == '0':
                _ue_transmission_power[u_i] = np.clip([10 ** (1/10.) * _ue_transmission_power[u_i]], 0, self.max_transmission_power)
            elif binary_action[action_searching_idx] == '1':
                _ue_transmission_power[u_i] = np.clip([10 ** (-1/10.) * _ue_transmission_power[u_i]], 0 , self.max_transmission_power)
            else:
                raise NotImplementedError
            action_searching_idx += 1

        # Beamforming Codebook Indices Control
        for u_i in range(self.num_ue):
            if binary_action[action_searching_idx] == '0':
                _ue_codebook_indices[u_i] = np.clip([_ue_codebook_indices[u_i]-1], 0, self.num_precoders-1)[0]
            elif binary_action[action_searching_idx] == '1':
                _ue_codebook_indices[u_i] = np.clip([_ue_codebook_indices[u_i]+1], 0, self.num_precoders-1)[0]
            else:
                raise NotImplementedError
            action_searching_idx += 1

        expected_next_state = np.zeros_like(state)
        expected_next_state[:4] = state[:4]
        expected_next_state[4:6] =_ue_transmission_power
        expected_next_state[6:8] =_ue_codebook_indices

        return expected_next_state

    def get_history_data(self,check_condition, data_storage_basepath=DATA_STORAGE):
        """

        Special case. Return only 2 users' SINR history data

        :param check_condition:
        :param data_storage_basepath:
        :return:
        """
        successful= True
        if check_condition == True:
            successful = self.reach_done and (np.sum(self.internal_reward_history) > 0) and (self.abort == False)

            # if self.train_epi_index < 250:
            #     successful = False

        if successful:
            dataset_storage = os.path.join(data_storage_basepath,self.config["config_name"])
            Path(dataset_storage).mkdir(parents=True, exist_ok=True)

            # np.save(os.path.join(dataset_storage, "ue_1_sinr.npy"), self._ue_SINR_queues[0])
            # np.save(os.path.join(dataset_storage, "ue_2_sinr.npy"), self._ue_SINR_queues[1])
            return self._ue_SINR_queues[0], self._ue_SINR_queues[1], self._ue_trasmission_power_queues[0], self._ue_trasmission_power_queues[1]
        else:
            return None, None, None, None