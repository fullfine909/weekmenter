import os
import csv
import glob
from my_utils.myutils import load_var
from weekmenter.main import get_image
from weekmenter.main import assign_numbers
from weekmenter.main import assign_days
from weekmenter.main import show_result
from weekmenter.segmentation import filter_data
from weekmenter.colors import apply_colors
from weekmenter.utils import get_mask_by_number

from weekmenter.const import IMAGE_SETS_PATH
from weekmenter.const import TEST_SETS_PATH
from weekmenter.const import TEST_SETS


def test_get_48_masks():
    # INIT TEST
    results = []
    errors = []
    # EXPECTED NUMBER OF MASKS
    ground_truth = {"S1": 48, "S2": 49}
    for test_set in TEST_SETS:
        files = glob.glob(os.path.join(TEST_SETS_PATH, test_set, "F*.bin"))
        files = sorted(files)
        for image_bin in files:
            image_name = os.path.basename(image_bin).split(".")[0]
            image = get_image(f"{IMAGE_SETS_PATH}/{test_set}/{image_name}.jpg")
            masks = load_var(image_bin)

            # PROCESS MASKS
            assign_numbers(masks)
            masks = filter_data(masks, image)
            days_dict = assign_days(masks)

            # CHECK FINAL NUMBER OF MASKS
            total_masks = sum([len(days_dict[day]) for day in days_dict])
            results.append({image_name: total_masks})
            if total_masks != ground_truth[test_set]:
                errors.append({image_name: total_masks, "set": test_set})

    assert len(errors) == 0


def test_get_colors():
    # INIT TEST
    results = []
    errors = []

    for test_set in TEST_SETS:
        # READ GROUND TRUTH COLORS
        colors_file = f"{TEST_SETS_PATH}/{test_set}/colors.csv"
        with open(colors_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            colors_table = list(reader)

        files = glob.glob(os.path.join(TEST_SETS_PATH, test_set, "F*.bin"))
        for image_bin in files:
            image_name = os.path.basename(image_bin).split(".")[0]
            image_path = f"{IMAGE_SETS_PATH}/{test_set}/{image_name}.jpg"
            image = get_image(image_path)
            masks = load_var(image_bin)

            # PROCESS MASKS
            assign_numbers(masks)
            masks = filter_data(masks, image)
            days_dict = assign_days(masks)
            apply_colors(masks)

            # APPLY COLORS
            for iday, day in enumerate(days_dict):
                for imask, mask in enumerate(days_dict[day]):
                    color = colors_table[imask][iday]
                    # CHECK FAULTY MASKS
                    mcolor = mask["color"]
                    if mcolor != color:
                        errors.append(
                            {
                                image_name: image_name,
                                "set": test_set,
                                "mcolor": mcolor,
                                "color": color,
                                "day": day,
                                "mask": imask,
                            }
                        )
                    results.append(mcolor == color)

    assert all(results)
