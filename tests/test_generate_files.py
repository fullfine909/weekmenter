import os
import csv
import glob
import numpy as np
from my_utils.myutils import load_var
from my_utils.myutils import save_var
from my_utils.myutils import flatten
from weekmenter.main import get_image
from weekmenter.main import assign_numbers
from weekmenter.main import assign_days
from weekmenter.main import apply_colors
from weekmenter.utils import get_mask_generator
from weekmenter.segmentation import filter_data
from weekmenter.colors import COLOR_NAMES as COLORS

from weekmenter.const import IMAGE_SETS_PATH
from weekmenter.const import TEST_SETS_PATH
from weekmenter.const import TEST_SETS
from weekmenter.const import COLORS_PATH
from weekmenter.const import COLORS_VAR


def test_generate_bin_files_with_masks():
    # ITERATE OVER TEST SETS
    for test_set in TEST_SETS:
        files = glob.glob(os.path.join(IMAGE_SETS_PATH, test_set, "*.jpg"))

        # ITERATE OVER IMAGES
        for image_path in files:
            # GET IMAGE NAME
            image_name = os.path.basename(image_path).split(".")[0]

            # GET MASKS
            if not glob.glob(f"{TEST_SETS_PATH}/{test_set}/{image_name}.bin"):
                image = get_image(image_path)
                mask_generator = get_mask_generator()
                masks = mask_generator.generate(image)
                bin_path = f"{TEST_SETS_PATH}/{test_set}/{image_name}"
                save_var(masks, bin_path)

    assert True


def test_generate_colors_dict():
    for test_set in TEST_SETS:
        # CHECK IF COLORS DICT ALREADY EXISTS
        colors_dict_path = f"{TEST_SETS_PATH}/{test_set}/colors_dict"
        if not glob.glob(f"{colors_dict_path}.bin"):
            # READ GROUND TRUTH COLORS
            colors_file = f"{TEST_SETS_PATH}/{test_set}/colors.csv"
            with open(colors_file, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                colors_table = list(reader)

            # CREATE A DICT TO SAVE PIXELS CORRESPONDING TO EACH COLOR
            colors_dict = {}
            for color in COLORS:
                colors_dict[color] = []

            # INIT TEST
            files = glob.glob(os.path.join(TEST_SETS_PATH, test_set, "F*.bin"))
            for image_bin in files:
                image_name = os.path.basename(image_bin).split(".")[0]
                image_path = f"{IMAGE_SETS_PATH}/{test_set}/{image_name}.jpg"
                image = get_image(image_path)
                masks = load_var(image_bin)

                # PROCESS MASKS
                assign_numbers(masks)
                masks = filter_data(masks)
                days_dict = assign_days(masks)

                # APPLY COLORS
                apply_colors(masks, image, extract_colors=False)
                for iday, day in enumerate(days_dict):
                    for imask, mask in enumerate(days_dict[day]):
                        pixels = mask["image_array"]
                        color = colors_table[imask][iday]
                        colors_dict[color].append(pixels)

            save_var(colors_dict, colors_dict_path)


def test_generate_colors():
    colors_set_dict = {}
    for test_set in TEST_SETS:
        colors_dict_bin_path = f"{TEST_SETS_PATH}/{test_set}/colors_dict"
        colors_dict = load_var(colors_dict_bin_path)
        colors_set_dict[test_set] = colors_dict

    # join the colors_set_dict into a single dict
    colors_joined = {}
    for key in colors_set_dict:
        for inner_key, value in colors_set_dict[key].items():
            if inner_key in colors_joined:
                colors_joined[inner_key] += value
            else:
                colors_joined[inner_key] = value

    # get the mean of each color
    colors_mean_dict = {}
    for color, color_array in colors_dict.items():
        cflat = flatten(color_array)
        cflatnp = np.array(cflat)
        colors_mean_dict[color] = tuple(np.mean(cflatnp, axis=0))

    save_var(colors_mean_dict, f"{COLORS_PATH}/{COLORS_VAR}")
