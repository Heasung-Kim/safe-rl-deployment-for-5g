import numpy as np
import os
import pickle
from global_config import ROOT_DIR
import copy


class GlobalTrainer(object):
    def __init__(self, config, env, policy):


        self.name = "None"

        self.config = config
        self._episode_length = config["episode_config"]["episode_length"]
        self._num_epi = config["episode_config"]["num_epi"]
        self.time_window = config["algorithm_config"]["time_window"]
        self._env = env
        self._policy = policy


    def __call__(self):
        total_steps = 0

        for epi_idx in range(self._num_epi):
            state = self._env.reset(epi_idx)
            epi_step = 0
            epi_total_reward = 0

            SINRs = []
            epi_info = []
            for _ in range(self._episode_length):
                total_steps += 1
                epi_step += 1
                if self._policy is not None:
                    action = self._policy.get_action()
                else:
                    action=15
                next_state, reward, done, info = self._env.step(action)
                #print(info["ue_serving_status"])
                SINRs.append(info["all_user_rate"])
                epi_info.append(copy.deepcopy(info))
                epi_total_reward += reward
                state = next_state
                if total_steps %10000 == 0 :
                    print("total_steps:", total_steps, ", current sum of rewards:", epi_total_reward)



            print("episode " , epi_idx ," average reward:", epi_total_reward/epi_step)


            self._save_epi_results(epi_info, epi_idx)

    def _save_epi_results(self, epi_info, epi_idx):

        data_storage_path = os.path.join(ROOT_DIR, "results", self.config["config_name"])
        if not os.path.exists(data_storage_path):
            os.makedirs(data_storage_path)

        algorithm_result_path = os.path.join(data_storage_path, self.config["algorithm_config"]["save_name"])
        if not os.path.exists(algorithm_result_path):
            os.makedirs(algorithm_result_path)


        epi_result_path = os.path.join(algorithm_result_path, "epi_" + str(epi_idx))
        if not os.path.exists(epi_result_path):
            os.makedirs(epi_result_path)

        with open(os.path.join(epi_result_path, 'rate_results.pickle'), 'wb') as handle:
            pickle.dump(epi_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("saved")
