import numpy as np
from weekmenter.colors import _classify_color
from weekmenter.results import show_result


# SEGMENT CLASSIFICATION
def filter_data(input_masks, image):
    masks = input_masks

    # Intial filters
    masks = _filter_by_ratio(masks)  # 1 < ratio < 3
    masks = _filter_by_shape(masks)  # width > height
    masks = _filter_by_area(masks, image)  # area < 0.01

    # Extract features (x, y, width, height, ratio, rgb)
    _extract_features(masks, image)

    # Filter: remove masks outside vertical lines (left and right) and horizontal line (bottom)
    masks, reference_masks = _filter_by_boundaries(masks, image)

    # Filter: remove masks which area is not similar to the reference masks
    masks = _filter_by_reference_masks(masks, reference_masks)

    # Filter: duplicates
    masks = _filter_duplicates(masks)
    # show_result(image, masks)
    return masks


def _filter_by_ratio(input_masks):
    masks = []
    for m in input_masks:
        ratio = m["bbox"][2] / m["bbox"][3]
        if ratio > 1 and ratio < 3:
            masks.append(m)
    return masks


def _filter_by_area(input_masks, image):
    masks = []
    image_area = image.shape[0] * image.shape[1]
    for m in input_masks:
        if m["area"] < image_area / 100:
            masks.append(m)
    return masks


def _filter_by_shape(input_masks):
    masks = []
    for m in input_masks:
        if m["bbox"][2] > m["bbox"][3]:  # width > height
            masks.append(m)
    return masks


def _extract_features(input_masks, image):
    masks = input_masks
    for m in masks:
        m["x"] = m["bbox"][0]
        m["y"] = m["bbox"][1]
        m["width"] = m["bbox"][2]
        m["height"] = m["bbox"][3]
        m["ratio"] = m["width"] / m["height"]
        m["rgb"] = _get_rgb_mean(m, image)
        # m["color"] = _classify_color(m["rgb"])
    return masks


def _get_rgb_mean(mask, image):
    width = image.shape[1]
    height = image.shape[0]
    mask_segmentation = mask["segmentation"]
    pixels = [image[i][j] for j in range(width) for i in range(height) if mask_segmentation[i][j]]
    rgb_mean = np.mean(pixels, axis=0)
    return rgb_mean


def _filter_by_boundaries(input_masks, image):
    left_b = _check_vertical_line(input_masks, start_from_left=True)
    right_b = _check_vertical_line(input_masks, start_from_left=False)
    bottom_y = _get_bottom_y(left_b, right_b)
    reference_masks = left_b + right_b

    masks = []
    for m in input_masks:
        cond1 = m["x"] >= left_b[1]["x"] and m["x"] <= right_b[1]["x"]
        cond2 = m["y"] <= bottom_y
        if cond1 and cond2:
            masks.append(m)
        else:
            print(f"Discarted mask {m['number']} (outside boundaries)")

    return masks, reference_masks


def _check_vertical_line(masks, N=3, start_from_left=True):
    # Sort the masks based on their x-coordinate
    masks.sort(key=lambda mask: mask["x"], reverse=not start_from_left)
    for i in range(len(masks) - N + 1):
        if _is_black(masks[i]):
            # if masks[i]["color"] == "BK":
            passable_masks = [masks[i]]
            j = i
            while True:
                current_mask = passable_masks[-1]
                next_mask = masks[j + 1]

                if start_from_left:
                    if current_mask["x"] + current_mask["width"] < next_mask["x"]:
                        break
                else:
                    if current_mask["x"] > next_mask["x"] + next_mask["width"]:
                        break

                if _is_black(next_mask):
                    # if next_mask["color"] == "BK":
                    passable_masks.append(next_mask)

                if len(passable_masks) == N:
                    if _check_passable_masks(passable_masks):
                        return passable_masks
                    break

                if j == len(masks) - N + 1:
                    break

                j += 1

    raise Exception(f"Could not find {N} passable masks")


def _get_bottom_y(left_b, right_b):
    masks = left_b + right_b
    masks = sorted(masks, key=lambda obj: obj["y"])
    bottom_y = masks[-1]["y"] + masks[-1]["height"]
    return bottom_y


def _check_passable_masks(masks):
    diff = []
    masks.sort(key=lambda mask: mask["y"])
    for i in range(len(masks) - 1):
        diff.append(abs(masks[i + 1]["y"] - masks[i]["y"]))
    mean_diff = np.mean(diff)
    margin = 0.25
    for d in diff:
        if mean_diff * (1 - margin) > d or d > mean_diff * (1 + margin):
            return False
    return True


def _filter_by_reference_masks(input_masks, reference_masks):
    marea = np.mean([m["area"] for m in reference_masks])
    masks = []
    margin = 0.5
    for m in input_masks:
        if marea * (1 - margin) < m["area"] < marea * (1 + margin):
            masks.append(m)
    return masks


def _is_black(mask):
    threshold = 80
    for rgb in mask["rgb"]:
        if rgb > threshold:
            return False
    return True


def _filter_duplicates(input_masks):
    masks = []
    for m in input_masks:
        if m["number"] not in [x["number"] for x in masks]:
            masks.append(m)
    return masks
