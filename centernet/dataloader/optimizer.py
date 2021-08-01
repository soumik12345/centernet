import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from .commons import project_to_2d
from ..commons import read_camera_intrinsic


class Optimizer:

    def __init__(
            self, image_shape: Tuple[int, int, int],
            image_height: int, image_width: int,
            model_scale: int, flipped: bool, dataset_path: str):
        self.image_shape = image_shape
        self.image_height = image_height
        self.image_width = image_width
        self.model_scale = model_scale
        self.flipped = flipped
        self.linear_model = LinearRegression()
        self.r, self.c = 0, 0
        (self.fx, self.fy, self.cx, self.cy), _, _ = read_camera_intrinsic(data_path=dataset_path)

    def fit_xyz(self, coordinates_dataframe: pd.DataFrame):
        x = coordinates_dataframe[['x', 'z']]
        y = coordinates_dataframe['y']
        self.linear_model.fit(x, y)
        print('MAE with x:', mean_absolute_error(y, self.linear_model.predict(x)))
        print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*self.linear_model.coef_))

    def distance_function(self, xyz):
        x, y, z = xyz
        xx = -x if self.flipped else x
        slope_err = (self.linear_model.predict([[xx, z]])[0] - y) ** 2
        x, y = project_to_2d(
            x, y, z,
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy
        )
        y, x = x, y
        x = (x - self.image_shape[0] // 2) * self.image_height \
            / (self.image_shape[0] // 2) / self.model_scale
        y = (y + self.image_shape[1] // 6) * self.image_width \
            / (self.image_shape[1] * 4 / 3) / self.model_scale
        return max(0.2, (x - self.r) ** 2 + (y - self.c) ** 2) + max(0.4, slope_err)

    def optimize(self, r, c, x0, y0, z0):
        self.r, self.c = r, c
        result = minimize(
            self.distance_function,
            np.array([x0, y0, z0]), method='Powell'
        )
        x_new, y_new, z_new = result.x
        return x_new, y_new, z_new
