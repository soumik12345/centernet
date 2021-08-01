import numpy as np
from math import sin, cos


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def preprocess_regression_targets(coordinate, flip: bool = False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            coordinate[k] = -coordinate[k]
    for name in ['x', 'y', 'z']:
        coordinate[name] = coordinate[name] / 100
    coordinate['roll'] = rotate(coordinate['roll'], np.pi)
    coordinate['pitch_sin'] = sin(coordinate['pitch'])
    coordinate['pitch_cos'] = cos(coordinate['pitch'])
    coordinate.pop('pitch')
    coordinate.pop('id')
    return coordinate


def project_to_2d(x, y, z, fx: float, fy: float, cx: float, cy: float):
    return x * fx / z + cx, y * fy / z + cy
