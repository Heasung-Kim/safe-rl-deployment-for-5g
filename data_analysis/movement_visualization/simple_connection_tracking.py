from matplotlib import animation
import math
from matplotlib.patches import RegularPolygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from global_config import ROOT_DIR
import os
import pickle
import yaml
import os
import time
import yaml
from env.user_equipment import UserEquipment

epi_name = "epi_0"
experiment_name = "draft_system"
data_storage_path = os.path.join(ROOT_DIR, "data", experiment_name)
results_data_storage_path = os.path.join(ROOT_DIR, "results", experiment_name, "pinv0")


os.chdir(ROOT_DIR)
paths = [os.path.join(ROOT_DIR, "figures", "bs.png")]
# UE Connection Information
with open(os.path.join(results_data_storage_path, epi_name, 'rate_results.pickle'), 'rb') as handle:
    epi_data = pickle.load(handle)

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=0.05)

if __name__ == '__main__':

    # get configuration
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    # test env settings
    num_ue = config["system_model_config"]["num_ue"]
    test_field_size = config["system_model_config"]["test_field_size"]
    cell_radius = config["system_model_config"]["cell_radius"]
    inter_site_distance = config["system_model_config"]["inter_site_distance"]

    # Initializing ues
    ues = [UserEquipment(config=config, ue_unique_num=i) for i in range(num_ue)]
    for i in range(num_ue):
        ues[i].set_scenario_data(epi_idx=0)
        #ues[i].set_scenario_mat_data(epi_idx=0)
        ues[i].result_analysis(epi_data=epi_data)


    # Drawing Setup
    fig = plt.figure()
    ax = plt.axes(xlim=(-test_field_size, test_field_size), ylim=(-test_field_size, test_field_size))

    ue_loc_plots = []
    for ue_idx in range(num_ue):
        ue = ues[ue_idx]
        ue_dot, = ax.plot([ue.x],[ue.y], 'o', color='lightgrey')
        ue_loc_plots.append(ue_dot)

    circle = plt.Circle((5, 5), 1, color='b', fill=False)
    ax.add_artist(circle)

    # Hexagon Drawing source code is from the following
    # https://stackoverflow.com/questions/46525981/how-to-plot-x-y-z-coordinates-in-the-shape-of-a-hexagonal-grid
    #

    BS_locations = [[0, 0], [0, inter_site_distance], [0, -inter_site_distance],
                    [inter_site_distance * np.cos(np.pi / 6), inter_site_distance * np.tan(np.pi / 6)], [2, -2]]
    coord = np.array([[0, 0, 0], [0, 1, -1],[1, 0, -1] ])#, [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0], ])
    colors = [["Black"], ["Red"], ["Orange"], ["DarkKhaki"], ["Green"], ["MediumBlue"], ["Violet"]][:3]
    small_colors = ["black", "red", "orange", "yellow", "green", "mediumblue", "violet"][:3]
    labels = [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6']][:3]

    # Horizontal cartesian coords
    hcoord = inter_site_distance * np.sqrt(3) / 2 * np.array([c[0] for c in coord])

    # Vertical cartersian coords
    vcoord = inter_site_distance * np.sqrt(3) / 2 * np.array(
        [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord])

    print("hcoord:", hcoord)
    print("vcoord:", vcoord)

    ax.set_aspect('equal')

    bs_ue_link_plots = []
    for i in range(num_ue):
        ln, = ax.plot([0,0], [0,0], 'b-', linewidth=0.5)
        bs_ue_link_plots.append(ln)

    # Drawing cells
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        color = c[0].lower()  # matplotlib understands lower case words for colours
        hex = RegularPolygon((x, y), numVertices=6, radius=inter_site_distance / 2.0 * 2.0 / np.sqrt(3),
                             orientation=np.radians(30),
                             facecolor=color, alpha=0.2, edgecolor='k')
        ax.add_patch(hex)
        # Also add a text label
        ax.text(x, y + 0.2, l[0], ha='center', va='center', size=15)

    ax.scatter(hcoord, vcoord, marker='d', c=[c[0].lower() for c in colors], alpha=0.3)
    ab = AnnotationBbox(getImage(paths[0]), (hcoord[0], vcoord[0]), frameon=False)
    ax.add_artist(ab)


    # animation function.  This is called sequentially
    def animate(frame):
        for ue in ues:
            ue.move()

        for ue_idx in range(num_ue):
            ue = ues[ue_idx]
            ue_loc_plots[ue_idx].set_data([ue.x], [ue.y])

            if ue.ue_connection_status != -1:
                color = small_colors[ue.ue_connection_status]
                ue_loc_plots[ue_idx].set_color(color)
            else:
                ue_loc_plots[ue_idx].set_color("lightgrey")

        current_time_idx = 0
        for ue_idx in range(num_ue):
            ue = ues[ue_idx]
            channels = ue.current_channel_for_all_bs
            channel_norm = np.linalg.norm(channels, axis=1)
            best_bs_idx = np.argmax(channel_norm)

            if ue.ue_serving_status > 0:
                bs_ue_link_plots[ue_idx].set_data([hcoord[ue.ue_connection_status], ue.x],
                                                  [vcoord[ue.ue_connection_status], ue.y])
                bs_ue_link_plots[ue_idx].set_color(colors[ue.ue_connection_status][0])
            else:
                bs_ue_link_plots[ue_idx].set_data([],
                                                  [])
            current_time_idx = ue.time_index
        #print(hcoord[0], ues[0].x, vcoord[0], ues[0].y)
        print(epi_data[current_time_idx]["ue_serving_status"])
        print("time idx", current_time_idx)
        return (*ue_loc_plots, *bs_ue_link_plots,)



    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=9999, interval=1, blit=True)
    # plt.grid()
    plt.xlabel("meters")
    plt.show()