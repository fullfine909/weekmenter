import numpy as np
from scipy.spatial import distance
from weekmenter.utils import get_colors_list


COLOR_NAMES = ["GR1", "GR2", "BL1", "BL2", "BL3", "BK", "YL", "PN", "PR", "OR"]
COLOR_LIST = get_colors_list()


def apply_colors(masks):
    for mask in masks:
        mask["color"] = _classify_color(mask["rgb"])


def _classify_color(rgb):
    predefined_colors = COLOR_LIST
    distances = [distance.euclidean(rgb, c) for c in predefined_colors.values()]
    closest_color_index = np.argmin(distances)
    closest_color_name = list(predefined_colors.keys())[closest_color_index]
    return closest_color_name
