import numpy as np
import matplotlib.pyplot as plt
from weekmenter.utils import get_mask_by_number


def show_result(image, masks, size=20, mask_index=None, mask_threshold=0.5):
    # plt.figure(figsize=(size,size))
    plt.imshow(image)
    if mask_index is None:
        _show_result(masks, mask_threshold)
    else:
        mask = [get_mask_by_number(masks, mask_index)]
        _show_result(mask, mask_threshold)
    plt.axis("off")
    plt.show()


def _show_result(masks, mask_threshold):
    if len(masks) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for idx, mask in enumerate(masks):
        m = mask["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        # Print the mask and the number of the mask
        ax.imshow(np.dstack((img, m * mask_threshold)))
        ax.text(
            mask["bbox"][0] + mask["bbox"][2] / 2,
            mask["bbox"][1] + mask["bbox"][3] / 2,
            str(mask["number"]),
            color="red",
        )
