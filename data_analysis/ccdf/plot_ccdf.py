# =====================================================================================================================
# Please note that part of this source codes used in this file is borrowed from Dr.Faris's project.
# We borrowed some part of the codes for plotting CCDF in
# https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks
#
# Mismar, Faris B., Brian L. Evans, and Ahmed Alkhateeb.
# "Deep reinforcement learning for 5G networks: Joint beamforming, power control, and interference coordination."
# IEEE Transactions on Communications 68.3 (2019): 1581-1592.
# =====================================================================================================================



import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from global_config import ROOT_DIR, DATA_STORAGE
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



def plot_ccdf(T, labels, ax ):
    num_bins = 10
    i = 0
    for data in T:
        data_ = T[data].dropna()

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        ccdf = np.insert(ccdf, 0, 1)
        bin_edges = np.insert(bin_edges[1:], 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
        lw = 1 + i

        style = '-'
        ax.plot(bin_edges, ccdf, style, linewidth=lw, label=labels[0])

    plt.grid(True)
    plt.tight_layout()
    ax.set_xlabel("Effective SINR (dB)")
    ax.set_ylabel('Complementary cumulative distribution')


##############################

def compute_distributions(algorithm_name):
    df_final = pd.DataFrame()
    gamma_0 = 5  # dB as in the environment file.
    start_epi = 0
    M=16
    num_epi = -1 #start_epi+10
    trial=0

    data1 = np.load(os.path.join(DATA_STORAGE, algorithm_name, "trial_" + str(trial) + "_" + 'ue_1_sinr.npy'),
                    allow_pickle=True)
    data1 = data1[start_epi:num_epi].flatten()
    df_1 = pd.DataFrame(data1).dropna()

    data2 = np.load(os.path.join(DATA_STORAGE, algorithm_name, "trial_" + str(trial) + "_" + 'ue_2_sinr.npy'),
                    allow_pickle=True)
    data2 = data2[start_epi:num_epi].flatten()
    df_2 = pd.DataFrame(data2).dropna()

    for trial in range(20):
        try:
            data1 = np.load(os.path.join(DATA_STORAGE, algorithm_name, "trial_"+str(trial)+"_"+'ue_1_sinr.npy'),allow_pickle=True)
            data1 = data1[start_epi:num_epi].flatten()
            df_1_add = pd.DataFrame(data1).dropna()
            df_1 = pd.concat([df_1,df_1_add])

            data2 = np.load(os.path.join(DATA_STORAGE, algorithm_name, "trial_"+str(trial)+"_"+'ue_2_sinr.npy'),allow_pickle=True)
            data2 = data2[start_epi:num_epi].flatten()
            df_2_add = pd.DataFrame(data2).dropna()
            df_2 = pd.concat([df_2,df_2_add])
        except:
            pass


    cutoff = gamma_0 + 10 * np.log2(M)
    df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    sinr = df.astype(float)
    indexes = (sinr <= cutoff) & ~np.isnan(sinr)
    df_sinr = pd.DataFrame(sinr[indexes])
    df_sinr.columns = ['sinr_{}'.format(M)]




    df_final = pd.concat([df_final, df_sinr], axis=1)

    return df_final



##############################
import yaml
if __name__ == '__main__':
    # get configuration
    with open(os.path.join(ROOT_DIR,"config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


    fig = plt.figure(figsize=(16,10))
    ax = fig.gca()


    df_final_ = compute_distributions(algorithm_name="optimal" )#config["algorithm_config"]["algorithm_name"])
    df_final = df_final_.values
    df_final = df_final.T
    sinr_16 = df_final
    sinr_16 = sinr_16[~np.isnan(sinr_16)]
    plot_ccdf(df_final_[['sinr_16']], labels=["Optimal"], ax=ax)


    df_final_ = compute_distributions(
        algorithm_name="SAC_discrete")  # config["algorithm_config"]["algorithm_name"])
    df_final = df_final_.values
    df_final = df_final.T
    sinr_16 = df_final
    sinr_16 = sinr_16[~np.isnan(sinr_16)]
    plot_ccdf(df_final_[['sinr_16']], labels=["SAC (discrete)"], ax=ax)

    df_final_ = compute_distributions(algorithm_name="bcmq_lr1e4_batch32_buffer10000")  # config["algorithm_config"]["algorithm_name"])
    df_final = df_final_.values
    df_final = df_final.T
    sinr_16 = df_final
    sinr_16 = sinr_16[~np.isnan(sinr_16)]
    plot_ccdf(df_final_[['sinr_16']], labels=["BCMQ (proposed)"], ax=ax)

    df_final_ = compute_distributions(
        algorithm_name="DQN")  # config["algorithm_config"]["algorithm_name"])
    df_final = df_final_.values
    df_final = df_final.T
    sinr_16 = df_final
    sinr_16 = sinr_16[~np.isnan(sinr_16)]
    plot_ccdf(df_final_[['sinr_16']], labels=["DQN"], ax=ax)




    plt.legend( loc="lower left")
    plt.show()