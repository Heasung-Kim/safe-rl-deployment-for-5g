import pandas as pd
import os
from global_config import ROOT_DIR, DATA_STORAGE
import math
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns;

#sns.set_theme()
sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=2.25)
plt.rcParams['font.size'] = '40'
from numpy.linalg import inv, norm
import yaml
import scipy.io as sio


projects_directory = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
results_save_path = os.path.join(projects_directory, "results")


def get_reward_history_data(algorithm_name, num_trial=20):
    data = []

    for i in range(num_trial):
        path = os.path.join(DATA_STORAGE, algorithm_name, "trial_"+str(i)+"_average_returns.npy")
        sample = np.load(path)
        data.append(sample)

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std


if __name__ == '__main__':
    algorithm_names = ["bcmq", "SAC_discrete", "DQN"]

    plt.figure( figsize=(16,10))
    plt.grid()

    bcmq_mean, bcmq_std = get_reward_history_data("bcmq_lr1e4_batch32_buffer10000")
    plt.fill_between(np.arange(len(bcmq_mean)), bcmq_mean - bcmq_std,
                     bcmq_mean + bcmq_std, alpha=0.1,
                     color="r")
    plt.plot(np.arange(len(bcmq_mean)), bcmq_mean, 'o-', color="r",
             label="BCMQ (proposed)")



    bcmq_mean, bcmq_std = get_reward_history_data("SAC_discrete")
    plt.fill_between(np.arange(len(bcmq_mean)), bcmq_mean - bcmq_std,
                     bcmq_mean + bcmq_std, alpha=0.1,
                     color="g")
    plt.plot(np.arange(len(bcmq_mean)), bcmq_mean, 'o-', color="g",
             label="Soft Actor-Critic (Discrete)")


    bcmq_mean, bcmq_std = get_reward_history_data("DQN")
    plt.fill_between(np.arange(len(bcmq_mean)), bcmq_mean - bcmq_std,
                     bcmq_mean + bcmq_std, alpha=0.1,
                     color="b")
    plt.plot(np.arange(len(bcmq_mean)), bcmq_mean, 'o-', color="b",
             label="DQN")



    plt.legend(loc="lower right")
    plt.xlabel("Learning Iteration (1e2)")
    plt.ylabel("Average Reward (over 1000 radio frames)")

    plt.show()
