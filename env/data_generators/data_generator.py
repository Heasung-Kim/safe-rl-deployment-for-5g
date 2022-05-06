from global_config import ROOT_DIR
import os
from abc import *
import yaml


class DataGenerator(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    def generate_data(self):
        data_storage_path = os.path.join(ROOT_DIR, "data", self.config["config_name"])
        if not os.path.exists(data_storage_path):
            os.makedirs(data_storage_path)

        with open(os.path.join(data_storage_path,'config.yaml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

        num_ue = int(self.config["system_model_config"]["num_ue"])
        num_epi = int(self.config["episode_config"]["num_epi"])
        for j in range(num_epi):
            epi_storage_path = os.path.join(data_storage_path, "epi_" + str(j))
            if not os.path.exists(epi_storage_path):
                os.makedirs(epi_storage_path)

            for i in range(num_ue):
                single_ue_data_storage_path = os.path.join(epi_storage_path, "ue_" + str(i))
                if not os.path.exists(single_ue_data_storage_path):
                    os.makedirs(single_ue_data_storage_path)
                self.generate_single_ue_data(path=single_ue_data_storage_path, ue_idx=i)

    @abstractmethod
    def generate_single_ue_data(self, path, ue_idx):
        pass

