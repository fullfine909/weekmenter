import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from my_utils.myutils import load_var
from weekmenter.const import COLORS_PATH, COLORS_VAR

# GLOBAL VARIABLES
MODEL_TYPE = "vit_l"
MODEL_DICT = {
    "vit_h": "sam_vit_h.pth",
    "vit_l": "sam_vit_l.pth",
    "vit_b": "sam_vit_b.pth",
}
CHECKPOINT_PATH = f"input/checkpoint/{MODEL_DICT[MODEL_TYPE]}"
VALID_DAYS = [2, 3, 4, 5, 6, 8, 9]


def get_mask_generator(model_type=MODEL_TYPE):
    sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def assign_numbers(masks):
    for idx, mask in enumerate(masks):
        mask["number"] = idx


def get_mask_by_number(masks, number):
    mask = [m for m in masks if m["number"] == number][0]
    return mask


def assign_days(masks):
    masks = sorted(masks, key=lambda x: x["x"])
    ticks = 10
    x0 = masks[1]["x"]
    x1 = masks[-2]["x"]
    step = (x1 - x0) / (ticks - 1)
    points = [x0 + step * i for i in range(ticks)]
    for mask in masks:
        scores = [abs(mask["x"] - point) for point in points]
        day = scores.index(min(scores)) + 1
        mask["day"] = day

    days_dict = {}
    for day in VALID_DAYS:
        days_dict[day] = []
        for mask in masks:
            if mask["day"] == day:
                days_dict[day].append(mask)

    for day in VALID_DAYS:
        day_values = sorted(days_dict[day], key=lambda x: x["y"])
        days_dict[day] = _filter_duplicates(day_values)

    return days_dict


def get_colors_list():
    try:
        colors_list = load_var(f"{COLORS_PATH}/{COLORS_VAR}")
    except FileNotFoundError:
        colors_list = {}
    return colors_list


def _filter_duplicates(day_values):
    filtered_day_values = []
    if day_values:
        for i in range(len(day_values) - 1):
            if abs(day_values[i]["y"] - day_values[i + 1]["y"]) > day_values[i]["height"] / 2:
                filtered_day_values.append(day_values[i])
        filtered_day_values.append(day_values[-1])
    return filtered_day_values
