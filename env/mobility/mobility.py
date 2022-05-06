import numpy as np
import yaml
from global_config import ROOT_DIR
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def _uniform_initialization(test_field_size):
    """

    :param test_field_size: -test_field_size ~ test_field_size is the horizontal/vertical length
    :return: initial user coordinate
    """
    x = test_field_size * (np.random.random() - 0.5) * 2
    y = test_field_size * (np.random.random() - 0.5) * 2
   
    return x, y

def _generate_random_mobility(config):
    return (np.random.random_sample() - 0.5) * config["system_model_config"]["ue_max_speed"]


def _generate_fixed_mobility(config):
    return config["system_model_config"]["ue_max_speed"]


def get_polygon_by_3_centers(config):
    bs_x_coordinate = config["system_model_config"]["bs_x_coordinate"]
    bs_y_coordinate = config["system_model_config"]["bs_y_coordinate"]
    coords = ((bs_x_coordinate[0], bs_y_coordinate[0]),
              (bs_x_coordinate[1], bs_y_coordinate[1]),
              (bs_x_coordinate[2], bs_y_coordinate[2]),
              (bs_x_coordinate[0], bs_y_coordinate[0]))
    polygon = Polygon(coords)

    return polygon

import math
def generate_random_ue_position_sequence(config, sequence_length, time_slot_length, init_coord=None, ue_idx=None):
    """

    :param config: System configuration
    :param sequence_length:
    :param time_slot_length: Time slot length. UE will move with velocity (m/s) * time_slot_length
    UE moves 100m per second. It will moves 100 * 0.04m per 1 * 0.04 sec.

    :return: two sequence_length size ndarray. (x, y coordinates)
    """

    x_coordinates = np.zeros(sequence_length)
    y_coordinates = np.zeros(sequence_length)
    test_field_size = config["system_model_config"]["test_field_size"]
    cell_radius = config["system_model_config"]["cell_radius"]
    initial_BS_x = config["system_model_config"]["bs_x_coordinate"][ue_idx]
    initial_BS_y = config["system_model_config"]["bs_y_coordinate"][ue_idx]


    center_polygon = get_polygon_by_3_centers(config)

    init_x, init_y = _uniform_initialization(test_field_size)
    if init_coord is not None:
        x_coordinates[0] = init_coord[0]
        y_coordinates[0] = init_coord[1]
    else:
        x_coordinates[0] = init_x
        y_coordinates[0] = init_y

    #velx = _generate_random_mobility(config)
    #vely = _generate_random_mobility(config)
    vel = _generate_fixed_mobility(config)
    for i in range(sequence_length - 1):

        theta_1 = np.random.uniform(low=-math.pi, high=math.pi, size=1)

        dx_1 = vel * math.cos(theta_1)
        dy_1 = vel * math.sin(theta_1)

        # Move UE 1
        x_coordinates[i + 1] = x_coordinates[i] + dx_1
        y_coordinates[i + 1] = y_coordinates[i] + dy_1

        """
        if np.random.random_sample() < 0.9995:
            x_coordinates[i + 1] = x_coordinates[i] + velx * time_slot_length
            y_coordinates[i + 1] = y_coordinates[i] + vely * time_slot_length
        else:
            velx = _generate_random_mobility(config)
            vely = _generate_random_mobility(config)
            x_coordinates[i + 1] = x_coordinates[i] + velx * time_slot_length
            y_coordinates[i + 1] = y_coordinates[i] + vely * time_slot_length
        """

        point = Point(x_coordinates[i + 1], y_coordinates[i + 1])  # create point
        polygon_contains_point = center_polygon.contains(point)  # check if polygon contains point
        #print(point.within(polygon))  # check if a point is in the polygon
        if polygon_contains_point is not True or \
                np.linalg.norm(np.array([x_coordinates[i + 1], y_coordinates[i+1]])-np.array([initial_BS_x,initial_BS_y]), ord=2)>= cell_radius*0.8:

            x_coordinates[i + 1] = x_coordinates[i]
            y_coordinates[i + 1] = y_coordinates[i]

            velx = _generate_random_mobility(config)
            vely = _generate_random_mobility(config)
    return np.array([x_coordinates, y_coordinates])

def _deprecated_generate_random_ue_position_sequence(config, sequence_length, time_slot_length, init_coord=None):
    """

    :param config: System configuration
    :param sequence_length:
    :param time_slot_length: Time slot length. UE will move with velocity (m/s) * time_slot_length
    UE moves 100m per second. It will moves 100 * 0.04m per 1 * 0.04 sec.

    :return: two sequence_length size ndarray. (x, y coordinates)
    """
    
    x_coordinates = np.zeros(sequence_length)
    y_coordinates = np.zeros(sequence_length)
    test_field_size = config["system_model_config"]["test_field_size"]

    init_x, init_y = _uniform_initialization(test_field_size)
    if init_coord is not None:
        x_coordinates[0] = init_coord[0]
        y_coordinates[0] = init_coord[1]
    else:
        x_coordinates[0] = init_x
        y_coordinates[0] = init_y

    
    velx = _generate_random_mobility(config)
    vely = _generate_random_mobility(config)
    
    for i in range(sequence_length-1):
        if np.random.random_sample() < 0.9995:
            x_coordinates[i+1] = x_coordinates[i] + velx * time_slot_length
            y_coordinates[i+1] = y_coordinates[i] + vely * time_slot_length
        else:
            velx = _generate_random_mobility(config)
            vely = _generate_random_mobility(config)
            x_coordinates[i+1] = x_coordinates[i] + velx * time_slot_length
            y_coordinates[i+1] = y_coordinates[i] + vely * time_slot_length
        if x_coordinates[i+1] >= test_field_size:
            x_coordinates[i+1] = test_field_size
            velx = -1 * velx
        if x_coordinates[i+1] <= -test_field_size:
            x_coordinates[i+1] = -test_field_size
            velx = -1 * velx
        if y_coordinates[i+1] >= test_field_size:
            y_coordinates[i+1] = test_field_size
            vely = -1 * vely
        if y_coordinates[i+1] <= -test_field_size:
            y_coordinates[i+1] = -test_field_size
            vely = -1 * vely
    return np.array([x_coordinates, y_coordinates])

if __name__ == '__main__':
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    coordinates = generate_random_ue_position_sequence(config=config, sequence_length=100, time_slot_length=0.1)
    print(coordinates)