from weekmenter.utils import get_mask_generator
from weekmenter.utils import assign_numbers
from weekmenter.utils import assign_days
from weekmenter.results import show_result
from weekmenter.segmentation import filter_data
from weekmenter.image import get_image
from weekmenter.colors import apply_colors

from weekmenter.const import IMAGE_SETS_PATH
from weekmenter.const import TEST_SETS

from my_utils.myutils import load_var, save_var


def get_initial_image(image_path):
    image = get_image(image_path)
    return image


def process_image(image):
    # GET MASKS
    # mask_generator = get_mask_generator()
    # masks = mask_generator.generate(image)
    # save_var(masks, "masks")
    masks = load_var("masks")

    # PROCESS MASKS
    assign_numbers(masks)
    masks = filter_data(masks, image)
    days_dict = assign_days(masks)
    apply_colors(masks)

    # show_result(image, masks)
    print(f"Number of masks: {len(masks)}")
    return len(masks)


def main():
    image_path = "image.jpg"
    image = get_initial_image(image_path)
    process_image(image)


if __name__ == "__main__":
    main()
