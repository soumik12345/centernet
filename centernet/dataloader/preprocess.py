import cv2
import numpy as np

from .commons import preprocess_regression_targets
from ..commons import string_to_coordinates, get_image_coordinates


class Preprocessor:

    def __init__(self, dataset_path: str, image_height: int, image_width: int, model_scale: int):
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width
        self.model_scale = model_scale

    def preprocess_image(self, image, flip=False):
        image = image[image.shape[0] // 2:]
        bg = np.ones_like(image) * image.mean(1, keepdims=True).astype(image.dtype)
        bg = bg[:, :image.shape[1] // 6]
        image = np.concatenate([bg, image, bg], 1)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = image[:, ::-1] if flip else image
        return (image / 255).astype('float32')

    def get_targets(self, image, labels, flip: bool):
        mask = np.zeros([
            self.image_height // self.model_scale,
            self.image_width // self.model_scale], dtype='float32'
        )
        coordinate_dataframe_fields = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        regression_targets = np.zeros([
            self.image_height // self.model_scale,
            self.image_width // self.model_scale, 7], dtype='float32'
        )
        coordinates = string_to_coordinates(labels)
        xs, ys, _ = get_image_coordinates(labels, dataset_path=self.dataset_path)
        for x, y, coordinate in zip(xs, ys, coordinates):
            x, y = y, x
            x = (x - image.shape[0] // 2) * self.image_height \
                / (image.shape[0] // 2) / self.model_scale
            x = np.round(x).astype('int')
            y = (y + image.shape[1] // 6) * self.image_width \
                / (image.shape[1] * 4 / 3) / self.model_scale
            y = np.round(y).astype('int')
            is_x_in_limit = 0 <= x < self.image_height // self.model_scale
            is_y_in_limit = 0 <= y < self.image_width // self.model_scale
            if is_x_in_limit and is_y_in_limit:
                mask[x, y] = 1
                coordinate = preprocess_regression_targets(coordinate, flip)
                regression_targets[x, y] = [coordinate[n] for n in sorted(coordinate)]
        if flip:
            mask = np.array(mask[:, ::-1])
            regression_targets = np.array(regression_targets[:, ::-1])
        return mask, regression_targets
