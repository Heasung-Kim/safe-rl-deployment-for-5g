from env.data_generators.data_generator import DataGenerator
from env.mobility.mobility import *
from env.channel.channel import *
import pickle
import os
import numpy as np
from scipy.ndimage.filters import uniform_filter1d


class Scenario(DataGenerator):
    """


    """
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config

    def generate_single_ue_data(self, path, ue_idx):
        config = self.config
        episode_length = int(config["episode_config"]["episode_length"])
        feedback_time_interval = config["system_model_config"]["feedback_time_interval"]

        num_bs = config["system_model_config"]["num_bs"]
        center_frequency = config["system_model_config"]["center_frequency"]
        prob_LOS = config["system_model_config"]["prob_LOS"]
        M_ULA = config["system_model_config"]["M_ULA"] # Tx antenna size

        # step1: generate ue coordinates
        init_coord = None
        if ue_idx==0:
            init_coord = [config["system_model_config"]["init_x_coordinate"][0], config["system_model_config"]["init_y_coordinate"][0]]
        elif ue_idx == 1:
            init_coord = [config["system_model_config"]["init_x_coordinate"][1], config["system_model_config"]["init_y_coordinate"][1]]
        elif ue_idx == 2:
            init_coord = [config["system_model_config"]["init_x_coordinate"][2], config["system_model_config"]["init_y_coordinate"][2]]
        ue_coordinates = generate_random_ue_position_sequence(
                                            config=config,
                                            sequence_length=episode_length,
                                            time_slot_length=feedback_time_interval,
                                            init_coord=init_coord, ue_idx=ue_idx)

        # step2: compute distance btw ue-BSs
        # num_of_bs * episode_length sized matrix


        # step3: generate channel
        # num_of_bs * Ntx * episode_length sized matrix
        channel_matrix = np.zeros(shape=(num_bs, episode_length, M_ULA), dtype=complex)
        for bs_i in range(num_bs):
            x_bs = config["system_model_config"]["bs_x_coordinate"][bs_i]
            y_bs = config["system_model_config"]["bs_y_coordinate"][bs_i]
            channel_value = None
            for epi_i in range(episode_length):
                x_ue = ue_coordinates[0][epi_i]
                y_ue = ue_coordinates[1][epi_i]
                new_channel_value = compute_channel(x_ue, y_ue, x_bs, y_bs, center_frequency, M_ULA, prob_LOS)

                #if epi_i == 0:
                channel_matrix[bs_i][epi_i] = new_channel_value
                #else:
                #    channel_matrix[bs_i][epi_i] = new_channel_value * alpha + channel_value * (1-alpha)
                channel_value = new_channel_value
            # c_real = uniform_filter1d(channel_matrix[bs_i].real, size=50)
            # c_imag = uniform_filter1d(channel_matrix[bs_i].imag, size= 50)
            # channel_matrix[bs_i] = np.array([1j]) * c_imag + c_real
        ue_data = {}
        ue_data["ue_coordinates"] = ue_coordinates
        ue_data["channel_matrix"] = channel_matrix

        with open(os.path.join(path, 'data.pickle'), 'wb') as handle:
            pickle.dump(ue_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("checkpoint")

    def generate_immediate_single_ue_data(self, ue_idx):
        config = self.config
        episode_length = int(config["episode_config"]["episode_length"])
        feedback_time_interval = config["system_model_config"]["feedback_time_interval"]

        num_bs = config["system_model_config"]["num_bs"]
        center_frequency = config["system_model_config"]["center_frequency"]
        prob_LOS = config["system_model_config"]["prob_LOS"]
        M_ULA = config["system_model_config"]["M_ULA"] # Tx antenna size

        # step1: generate ue coordinates
        init_coord = [0,0]
        init_coord[0] = config["system_model_config"]["bs_x_coordinate"][ue_idx]
        init_coord[1] = config["system_model_config"]["bs_y_coordinate"][ue_idx]
        np.random.seed()
        init_coord[0] += np.random.uniform(low=-self.config["system_model_config"]["cell_radius"], high=self.config["system_model_config"]["cell_radius"])
        init_coord[1] += np.random.uniform(low=-self.config["system_model_config"]["cell_radius"], high=self.config["system_model_config"]["cell_radius"])
        ue_coordinates = generate_random_ue_position_sequence(
                                            config=config,
                                            sequence_length=episode_length,
                                            time_slot_length=feedback_time_interval,
                                            init_coord=init_coord, ue_idx=ue_idx)

        # step2: compute distance btw ue-BSs
        # num_of_bs * episode_length sized matrix


        # step3: generate channel
        # num_of_bs * Ntx * episode_length sized matrix
        channel_matrix = np.zeros(shape=(num_bs, episode_length, M_ULA), dtype=complex)
        for bs_i in range(num_bs):
            x_bs = config["system_model_config"]["bs_x_coordinate"][bs_i]
            y_bs = config["system_model_config"]["bs_y_coordinate"][bs_i]
            channel_value = None
            for epi_i in range(episode_length):
                x_ue = ue_coordinates[0][epi_i]
                y_ue = ue_coordinates[1][epi_i]
                new_channel_value = compute_channel(x_ue, y_ue, x_bs, y_bs, center_frequency, M_ULA, prob_LOS)

                #if epi_i == 0:
                channel_matrix[bs_i][epi_i] = new_channel_value
                #else:
                #    channel_matrix[bs_i][epi_i] = new_channel_value * alpha + channel_value * (1-alpha)
                channel_value = new_channel_value
            # c_real = uniform_filter1d(channel_matrix[bs_i].real, size=50)
            # c_imag = uniform_filter1d(channel_matrix[bs_i].imag, size= 50)
            # channel_matrix[bs_i] = np.array([1j]) * c_imag + c_real
        ue_data = {}
        ue_data["ue_coordinates"] = ue_coordinates
        ue_data["channel_matrix"] = channel_matrix

        return ue_data


if __name__ == '__main__':
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    my_scene = Scenario(config)
    my_scene.generate_data()