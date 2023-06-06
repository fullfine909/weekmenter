import cv2


DEFAULT_SIZE = (800, 600)


def get_image(image_path):
    img = cv2.imread(image_path)
    img = _process_image(img)
    return img


def _process_image(input_image, size=DEFAULT_SIZE, display=False):
    img = input_image.copy()

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define the new size
    new_size = size

    # Resize the image while maintaining the aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]
    new_width = int(new_size[0] * aspect_ratio) if aspect_ratio > 1 else new_size[0]
    new_height = int(new_size[1] / aspect_ratio) if aspect_ratio <= 1 else new_size[1]
    img = cv2.resize(img, (new_width, new_height))

    # Display the image
    if display:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img
