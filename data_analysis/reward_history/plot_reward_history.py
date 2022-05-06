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

def get_reward_history_data(algorithm_name, df, num_trial=20):
    path = os.path.join(DATA_STORAGE, algorithm_name, "average_returns.npy")
    data = np.load(path)
    length = len(data)
    if df is None:
        df = pd.DataFrame({'sample_index': np.arange(length), })

    df[algorithm_name] = data
    return df


if __name__ == '__main__':
    algorithm_names = ["bcmq", "SAC_discrete", "DQN"]
    df=None
    for algorithm_name in algorithm_names:
        df = get_reward_history_data(algorithm_name, df)


    df = df.drop(columns=['sample_index'])
    # plt.figure()
    heading_properties = [('font-size', '1000px')]

    cell_properties = [('font-size', '50px')]

    dfstyle = [dict(selector="th", props=heading_properties),\
     dict(selector="td", props=cell_properties)]

    df.style.set_table_styles(dfstyle)
    ax = df.plot(use_index=False, markevery=1000, markersize=8, figsize=(16,10))
    markers = ['H', '^', 'v', 's', '3', '.', '1', '_']
    linestyles = ['-', '--', '-.', ':', '', ' ', 'None', ]
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])
        #line.set_linestyle(linestyles[i])
    ax.legend(ax.get_lines(), df.columns, loc='best')
    ax.set_ylabel("Reward (1000 epi Average)")
    ax.set_xlabel("iteration (1e3)")
    #plt.savefig(os.path.join(ROOT_DIR,"data_analysis", "plots", "achievable_rate", "fig_obs_noise_difference_"+basename+str(feedback_bit)+".eps"), bbox_inches='tight')
    plt.show(block=True)

