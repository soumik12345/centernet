import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from ..commons import read_camera_intrinsic
from .commons import post_process_regression, project_to_2d


class PostProcess:

    def __init__(
            self, image_shape: Tuple[int, int, int], image_height,
            image_width, model_scale, distance_threshold: int, dataset_path: str):
        self.image_shape = image_shape
        self.image_height = image_height
        self.image_width = image_width
        self.model_scale = model_scale
        self.distance_threshold = distance_threshold
        (self.fx, self.fy, self.cx, self.cy), self.camera_matrix, _ = read_camera_intrinsic(data_path=dataset_path)
        self.coordinate_fit = LinearRegression()

    def fit_coordinates(self, coordinates_dataframe: pd.DataFrame):
        x = coordinates_dataframe[['x', 'z']]
        y = coordinates_dataframe['y']
        self.coordinate_fit.fit(x, y)
        # print('MAE with x:', mean_absolute_error(y, self.coordinate_fit.predict(x)))
        # print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*self.coordinate_fit.coef_))

    def optimize_xy(self, r, c, x0, y0, z0, flipped=False):
        def distance_fn(xyz):
            x, y, z = xyz
            xx = -x if flipped else x
            slope_err = (self.coordinate_fit.predict([[xx, z]])[0] - y) ** 2
            x, y = project_to_2d(x, y, z, self.fx, self.fy, self.cx, self.cy)
            y, x = x, y
            x = (x - self.image_shape[0] // 2) * self.image_height \
                / (self.image_shape[0] // 2) / self.model_scale
            y = (y + self.image_shape[1] // 6) * self.image_width \
                / (self.image_shape[1] * 4 / 3) / self.model_scale
            return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)
        res = minimize(distance_fn, np.array([x0, y0, z0]), method='Powell')
        x_new, y_new, z_new = res.x
        return x_new, y_new, z_new

    def clear_duplicates(self, coordinates):
        for c1 in coordinates:
            xyz1 = np.array([c1['x'], c1['y'], c1['z']])
            for c2 in coordinates:
                xyz2 = np.array([c2['x'], c2['y'], c2['z']])
                distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
                if distance < self.distance_threshold:
                    if c1['confidence'] < c2['confidence']:
                        c1['confidence'] = -1
        return [c for c in coordinates if c['confidence'] > 0]

    def extract_coordinates(self, prediction, flipped=False):
        logits = prediction[0]
        regression_output = prediction[1:]
        points = np.argwhere(logits > 0)
        col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
        coordinates = []
        for r, c in points:
            regression_dict = dict(zip(col_names, regression_output[:, r, c]))
            coordinates.append(post_process_regression(regression_dict))
            coordinates[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
            coordinates[-1]['x'], coordinates[-1]['y'], coordinates[-1]['z'] = \
                self.optimize_xy(
                    r, c,
                    coordinates[-1]['x'],
                    coordinates[-1]['y'],
                    coordinates[-1]['z'], flipped
                )
        return self.clear_duplicates(coordinates)
