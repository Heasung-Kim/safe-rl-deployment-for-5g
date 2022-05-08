from global_config import ROOT_DIR, DATA_STORAGE, ALGORITHMIC_DATA_STORAGE
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch
import copy
import yaml
from algorithms.replay_buffer import Replay_Buffer
from algorithms.BCMQ import BCMQ
from pathlib import Path
from runners.utils import save_algorithm_result
from global_config import ROOT_DIR, DATA_STORAGE
from env.dynamic_envinronment import Environment

def get_constrained_randomly_generated_batch(config, replay_buffer):
    buffer_max_size = config["algorithm_config"]["buffer_max_size"]
    epi_length_max = config["episode_config"]["episode_length"]
    cnt = 0
    while(cnt < buffer_max_size):
        state = env.reset()
        for t in range(int(epi_length_max)):
            if config["algorithm_config"]["bs_coordination_level"] == 3:
                action = env.action_space.sample()
            elif config["algorithm_config"]["bs_coordination_level"] == 2:
                action = np.random.randint(0,3,size=1)[0]
                if np.random.rand() < 0.2:
                    action = env.action_space.sample()
            else:                # No coordination
                action = np.random.randint(0,3,size=1)[0]
            next_state, reward, done, info = env.step(action)
            # Store data in replay buffer
            replay_buffer.add_sample(state, action, next_state, reward, done*1.0)
            cnt +=1
            state = copy.copy(next_state)
    replay_buffer.save()

def train(config, env, replay_buffer, device):

    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n

    policy = BCMQ(config,
        action_shape,
        state_shape,
        device
    )

    replay_buffer.load()

    average_returns = []
    SINR_1_historys = []
    SINR_2_historys = []
    transmission_power_1_historys = []
    transmission_power_2_historys = []

    algorithm_name = config["algorithm_config"]["algorithm_name"]
    max_num_steps = config["algorithm_evaluation_config"]["num_max_steps"]
    test_interval = config["algorithm_evaluation_config"]["test_interval"]
    for i in range(max_num_steps):
        if i % test_interval == 0:
            print("Current Learning step: ", i , "//", max_num_steps)
            evaluation, SINR_1_history, SINR_2_history, transmission_power_1_history, transmission_power_2_history = test(config, policy)
            average_returns.append(evaluation)
            if len(SINR_1_history) > 0:
                SINR_1_historys.extend(SINR_1_history)
                SINR_2_historys.extend(SINR_2_history)
                transmission_power_1_historys.extend(transmission_power_1_history)
                transmission_power_2_historys.extend(transmission_power_2_history)
        policy.train(replay_buffer)
    Path(os.path.join(DATA_STORAGE, algorithm_name)).mkdir(parents=True, exist_ok=True)
    save_algorithm_result(os.path.join(DATA_STORAGE, algorithm_name), average_returns, SINR_1_historys, SINR_2_historys,transmission_power_1_historys, transmission_power_2_historys)

def test(config, policy):

    eval_env = Environment(config=config, TRAIN_MODE=False)
    test_episodes = config["algorithm_evaluation_config"]["test_episodes"]

    average_return = 0.

    SINR_1_historys = []
    SINR_2_historys = []
    transmission_power_1_historys = []
    transmission_power_2_historys = []

    for i in range(test_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_model_based_rollout_action(np.array(state), env=eval_env)
            state, reward, done, info = eval_env.step(action)
            average_return += reward
        SINR_1_history, SINR_2_history, transmission_power_1_history, transmission_power_2_history = eval_env.get_history_data(check_condition=True,
                                                                    data_storage_basepath=DATA_STORAGE)
        if SINR_1_history is not None:
            SINR_1_historys.append(list(SINR_1_history))
            SINR_2_historys.append(list(SINR_2_history))
            transmission_power_1_historys.append(list(transmission_power_1_history))
            transmission_power_2_historys.append(list(transmission_power_2_history))
            print("SAVE SUCCESSFUL HISTORY idx:", len(SINR_1_historys))
    average_return /= test_episodes

    print("---------------------------------------")
    print("Test Performance: ", average_return)
    print("---------------------------------------")
    return average_return, SINR_1_historys, SINR_2_historys, transmission_power_1_historys, transmission_power_2_historys


if __name__ == "__main__":
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    env = Environment(config=config, TRAIN_MODE=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    for num_trial in range(config["algorithm_config"]["trial"]):
        print("==================================================")
        print("==================================================")
        print("=============trial "+str(num_trial) +"=================")
        print("==================================================")
        print("==================================================")
        replay_buffer = Replay_Buffer(config, state_shape, action_shape, ALGORITHMIC_DATA_STORAGE)
        get_constrained_randomly_generated_batch(config, replay_buffer)
        train(config, env, replay_buffer, device)