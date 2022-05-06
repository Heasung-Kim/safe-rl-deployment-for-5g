import os
import yaml
from global_config import ROOT_DIR,DATA_STORAGE
import pandas as pd
import os
from global_config import ROOT_DIR
import math
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

#sns.set_theme()
sns.set(rc={'figure.figsize':(16,10)})
sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=2.25)
plt.rcParams['font.size'] = '40'
import yaml
from env.channel.channel import compute_channel

def get_channel_realization(config, num_realization):
    bs_x_coordinate = config["system_model_config"]["bs_x_coordinate"]
    bs_y_coordinate = config["system_model_config"]["bs_y_coordinate"]

    init_x_coordinate = config["system_model_config"]["init_x_coordinate"]
    init_y_coordinate = config["system_model_config"]["init_y_coordinate"]
    f_c = config["system_model_config"]["center_frequency"]
    M_ULA = config["system_model_config"]["M_ULA"]
    prob_LOS = config["system_model_config"]["prob_LOS"]

    x_ue = init_x_coordinate[0]
    y_ue = init_y_coordinate[0]
    x_bs= bs_x_coordinate[0]
    y_bs= bs_y_coordinate[0]

    channel_realization = np.zeros(shape=(num_realization, M_ULA))
    for i in range(num_realization):
        channel = compute_channel(x_ue, y_ue, x_bs, y_bs, f_c, M_ULA, prob_LOS)
        channel_realization[i] = channel
    return channel_realization

if __name__ == '__main__':
    # get configuration
    with open(os.path.join(ROOT_DIR,"config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    channel_realizations = get_channel_realization(config, num_realization=10000)

    import numpy as np
    from sklearn.manifold import TSNE
    X = channel_realizations #np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random').fit_transform(X)
    print(X_embedded.shape)
    ax = sns.scatterplot(x=X_embedded[:,0], y= X_embedded[:,1],
                         label="T-SNE Geometrical Channel Model", edgecolor='none', color="k",
                         alpha=.3)  # , hue=frame["Object Name"])
    ax.set_xlabel("Reduced Latent 01")
    ax.set_ylabel("Reduced Latent 02")
    plt.savefig(os.path.join(ROOT_DIR, "data", "figures","TSNE_channel_model.jpg"), bbox_inches='tight', transparent=True)
    plt.show()