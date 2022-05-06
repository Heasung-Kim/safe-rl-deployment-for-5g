import fnmatch
import os
import numpy as np

def save_algorithm_result(data_storage_path,average_returns, SINR_1_historys, SINR_2_historys,transmission_power_1_historys, transmission_power_2_historys):

    num_trial = len(fnmatch.filter(os.listdir(data_storage_path), '*.npy'))
    num_trial = int(num_trial/5)


    np.save(os.path.join(data_storage_path, "trial_" + str(num_trial) + "_average_returns.npy"), np.array(average_returns))
    np.save(os.path.join(data_storage_path, "trial_" + str(num_trial) + "_ue_1_sinr.npy"), np.array(SINR_1_historys))
    np.save(os.path.join(data_storage_path, "trial_" + str(num_trial) + "_ue_2_sinr.npy"), np.array(SINR_2_historys))


    np.save(os.path.join(data_storage_path, "trial_" + str(num_trial) + "_ue_1_trasmission_power.npy"), np.array(transmission_power_1_historys))
    np.save(os.path.join(data_storage_path, "trial_" + str(num_trial) + "_ue_2_trasmission_power.npy"), np.array(transmission_power_2_historys))

    return