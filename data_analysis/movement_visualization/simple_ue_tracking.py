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
from env.user_equipment import UserEquipment

data_storage_path = os.path.join(ROOT_DIR, "data", "cost2100s1")
map_size = 400
N = 30

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=0.08)
paths = [
    os.path.join(ROOT_DIR, "figures", "bs.png")]

with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as stream:
    try:
        config = yaml.safe_load(stream)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

# Initializing ues
ues = [UserEquipment(config=config, ue_unique_num=i+1) for i in range(N)]
for i in range(N):
    ues[i].set_scenario_mat_data(epi_idx=1)

# First set up the figure, the axis, and the data_analysis element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-map_size, map_size), ylim=(-map_size, map_size))
d, = ax.plot([dot.x for dot in ues],
             [dot.y for dot in ues], 'ro')
circle = plt.Circle((5, 5), 1, color='b', fill=False)
ax.add_artist(circle)



cell_radius = 150  # in meters.
inter_site_distance = 3 * cell_radius / 2.
BS_locations = [[0, 0], [0, inter_site_distance], [0, -inter_site_distance],
                    [inter_site_distance * np.cos(np.pi / 6), inter_site_distance * np.tan(np.pi / 6)], [2, -2]]
coord = np.array([[0, 0, 0], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0], [1, 0, -1]])
colors = [["Black"], ["Red"], ["Orange"], ["DarkKhaki"], ["Green"], ["MediumBlue"], ["Violet"]]
small_colors = ["black", "red", "orange", "yellow", "green", "mediumblue", "violet"]
labels = [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6']]

# Horizontal cartesian coords
hcoord = inter_site_distance  * np.sqrt(3)/2* np.array([c[0] for c in coord])

# Vertical cartersian coords
vcoord = inter_site_distance * np.sqrt(3)/2* np.array([2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord])

ax.set_aspect('equal')

bs_ue_link_plots = []
for i in range(N):
    ln, = ax.plot([], [], 'b-', linewidth=0.5)
    bs_ue_link_plots.append(ln)

# Add some coloured hexagons
for x, y, c, l in zip(hcoord, vcoord, colors, labels):
    color = c[0].lower()  # matplotlib understands lower case words for colours
    hex = RegularPolygon((x, y), numVertices=6, radius=inter_site_distance/2.0 * 2.0 / np.sqrt(3),
                         orientation=np.radians(30),
                         facecolor=color, alpha=0.2, edgecolor='k')
    ax.add_patch(hex)
    # Also add a text label
    ax.text(x, y+0.2, l[0], ha='center', va='center', size=20)

ax.scatter(hcoord, vcoord, c=[c[0].lower() for c in colors], alpha=0.5)
ab = AnnotationBbox(getImage(paths[0]), (hcoord[0], vcoord[0]), frameon=False)
ax.add_artist(ab)

print("hc:", hcoord)
print("vc:", vcoord)

# animation function.  This is called sequentially
def animate(frame):
    for ue in ues:
        ue.move()
    d.set_data([dot.x for dot in ues],
               [dot.y for dot in ues])

    for ue_idx in range(N):
        channels = ues[ue_idx].current_channel_for_all_bs
        channel_norm = np.linalg.norm(channels, axis=1)
        best_bs_idx = np.argmax(channel_norm)
        bs_ue_link_plots[ue_idx].set_data([hcoord[best_bs_idx], ues[ue_idx].x],[vcoord[best_bs_idx], ues[ue_idx].y])

    print(hcoord[0], ues[0].x, vcoord[0], ues[0].y)
    return (d, *bs_ue_link_plots,)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=9999, interval=1)#, blit=True)
#plt.grid()
plt.show()
