import numpy as np
import cv2


def random_crop(images, crop_size=(192, 192)):
    """
        args:
        images: numpy.array (M,C,H,W)
        crop_size: crop size (e.g. 192x192)
        returns np.array (M,H,W,C)
    """
    m, c, h, w = images.shape
    assert h >= crop_size[0] and w >= crop_size[1], 'The crop size should smaller than image size'
    h1 = np.random.randint(0, h - crop_size[0] + 1)
    w1 = np.random.randint(0, w - crop_size[1] + 1)

    result_images = np.empty((m, c, h, w), dtype=images.dtype)
    for i, image in enumerate(images):
        crop_image = image[:, h1:h1 + crop_size[0], w1:w1 + crop_size[1]]  # CxHxw
        crop_image = cv2.resize(crop_image.transpose(1, 2, 0), dsize=(w, h))  # HxWxC
        if len(crop_image.shape) == 2:
            crop_image = np.expand_dims(crop_image, axis=-1)
        result_images[i] = crop_image.transpose(2, 0, 1)  # CxHxW
    return result_images


def random_cutout(images, min_cut=10, max_cut=30):
    """
        args:
        images: np.array shape (M,C,H,W)
        min / max cut: int, min / max size of cutout
        returns np.array (M,C,H,W)
    """

    m, c, h, w = images.shape
    cutout_size = np.random.randint(min_cut, max_cut)
    w1 = np.random.randint(0, w - cutout_size + 1)
    h1 = np.random.randint(min_cut, h - cutout_size + 1)
    images[:, :, h1:h1 + cutout_size, w1:w1 + cutout_size] = 0
    return images


def random_cutout_color(images, min_cut=10, max_cut=30):
    """
        args:
        images: np.array shape (M,C,H,W)
        min / max cut: int, min / max size of cutout
        returns np.array (M,C,H,W)
    """

    m, c, h, w = images.shape
    cutout_size = np.random.randint(min_cut, max_cut)
    w1 = np.random.randint(0, w - cutout_size + 1)
    h1 = np.random.randint(min_cut, h - cutout_size + 1)
    rand_color = np.random.randint(0, 255) / 255.
    images[:, :, h1:h1 + cutout_size, w1:w1 + cutout_size] = rand_color

    return images


# random flip
def random_flip(images, p=.2):
    """
        args:
        images: numpy.array shape (M,C,H,W)
        p: prob of applying aug,
        returns numpy.array shape (M,C,H,W)
    """
    rand_p = np.random.rand()
    if rand_p < p:
        flipped_images = np.flip(images, axis=3)  # horizontal flip
    else:
        flipped_images = images

    if rand_p > 1 - p:
        flipped_images = np.flip(flipped_images, axis=2)  # vertical flip
    else:
        flipped_images = flipped_images

    return flipped_images


# random rotation
def random_rotation(images):
    """
        args:
        images: numpy.array shape (M,C,H,W)
        p: prob of applying aug,
        returns numpy.array shape (M,C,H,W)
    """
    rotation_type = np.random.randint(0, 4)
    rotation_images = np.rot90(images, k=rotation_type, axes=(2, 3))
    return rotation_images


# random sensor noise
def random_sensor_noise(vector: np.ndarray, mean=0, sigma=0.5):
    noise = np.random.randn(*vector.shape) * sigma + mean
    return vector + noise


def no_aug(x):
    return x
