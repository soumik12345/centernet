import os
import cv2
import numpy as np
from typing import Tuple
from math import sin, cos

import pandas as pd


def read_camera_intrinsic(data_path) -> Tuple[Tuple[float, float, float, float], np.array, np.array]:
    camera_file = open(os.path.join(data_path, 'camera/camera_intrinsic.txt'), 'r')
    camera_file_lines = camera_file.readlines()
    fx, fy, cx, cy = [line.split('=')[-1].strip()[:-1] for line in camera_file_lines]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    camera_matrix_inverse = np.linalg.inv(camera_matrix)
    return (fx, fy, cx, cy), camera_matrix, camera_matrix_inverse


def string_to_coordinates(string):
    coordinates, names = [], ['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']
    for _ in np.array(string.split()).reshape([-1, 7]):
        coordinates.append(dict(zip(names, _.astype('float'))))
        if 'id' in coordinates[-1]:
            coordinates[-1]['id'] = int(coordinates[-1]['id'])
    return coordinates


def get_coordinates_dataframe(dataframe: pd.DataFrame):
    coordinates_dataframe = pd.DataFrame()
    for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
        arr = []
        for prediction_string in dataframe['PredictionString']:
            coordinates = string_to_coordinates(prediction_string)
            arr += [c[col] for c in coordinates]
        coordinates_dataframe[col] = arr
    return coordinates_dataframe


def get_image_coordinates(prediction_string, dataset_path: str):
    _, camera_matrix, _ = read_camera_intrinsic(dataset_path)
    coordinates = string_to_coordinates(prediction_string)
    xs = [c['x'] for c in coordinates]
    ys = [c['y'] for c in coordinates]
    zs = [c['z'] for c in coordinates]
    ground_truth = np.array(list(zip(xs, ys, zs))).T
    image_gt = np.dot(camera_matrix, ground_truth).T
    image_gt[:, 0] /= image_gt[:, 2]
    image_gt[:, 1] /= image_gt[:, 2]
    image_xs = image_gt[:, 0]
    image_ys = image_gt[:, 1]
    # z = Distance from the camera
    image_zs = image_gt[:, 2]
    return image_xs, image_ys, image_zs


def euler_angel_to_rotation_matrix(yaw, pitch, roll):
    y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    p = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    r = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    return np.dot(y, np.dot(p, r))


def draw_line(image, points, color: Tuple[int, int, int] = (255, 0, 0)):
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image


def visualize_in_3d(image, coordinates, camera_matrix: np.array):
    x_l, y_l, z_l = 1.02, 0.80, 2.31
    image = image.copy()
    for point in coordinates:
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        rotation_matrix = np.eye(4)
        t = np.array([x, y, z])
        rotation_matrix[:3, 3] = t
        rotation_matrix[:3, :3] = euler_angel_to_rotation_matrix(yaw, pitch, roll).T
        rotation_matrix = rotation_matrix[:3, :]
        p = np.array([
            [x_l, -y_l, -z_l, 1],
            [x_l, -y_l, z_l, 1],
            [-x_l, -y_l, z_l, 1],
            [-x_l, -y_l, -z_l, 1],
            [0, 0, 0, 1]
        ]).T
        image_cor_points = np.dot(camera_matrix, np.dot(rotation_matrix, p))
        image_cor_points = image_cor_points.T
        image_cor_points[:, 0] /= image_cor_points[:, 2]
        image_cor_points[:, 1] /= image_cor_points[:, 2]
        image_cor_points = image_cor_points.astype(int)
        image = draw_line(image, image_cor_points)
        image = draw_points(image, image_cor_points[-1:])
    return image
