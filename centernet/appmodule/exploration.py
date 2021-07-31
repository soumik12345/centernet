import os
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt

from .commons import add_heading
from ..commons import (
    get_coordinates_dataframe, get_image_coordinates,
    visualize_in_3d, string_to_coordinates, read_camera_intrinsic
)


def _show_car_coordinates(
        dataframe: pd.DataFrame, n_samples: int, dataset_path: str):
    image_ids = list(dataframe['ImageId'])
    prediction_strings = list(dataframe['PredictionString'])
    for index in range(n_samples):
        col1, col2 = st.beta_columns(2)
        image = Image.open(
            os.path.join(
                dataset_path,
                'train_images/{}.jpg'.format(image_ids[index])
            )
        )
        with col1:
            plt.figure(figsize=(18, 18))
            plt.imshow(image)
            plt.title(image_ids[index])
            plt.axis('off')
            st.pyplot(plt)
        x, y, _ = get_image_coordinates(
            prediction_strings[index], dataset_path=dataset_path
        )
        with col2:
            plt.figure(figsize=(14, 14))
            plt.imshow(image)
            plt.scatter(x, y, color='red', s=100)
            plt.title(image_ids[index] + '_labelled')
            plt.axis('off')
            st.pyplot(plt)


def _show_points_distribution(
        sample_image_id: str, dataframe: pd.DataFrame, dataset_path: str):
    xs, ys = [], []
    for points in dataframe['PredictionString']:
        x, y, _ = get_image_coordinates(points, dataset_path=dataset_path)
        xs += list(x)
        ys += list(y)
    plt.figure(figsize=(18, 18))
    plt.imshow(Image.open(os.path.join(
                dataset_path + 'train_images/{}.jpg'.format(sample_image_id))), alpha=0.3)
    plt.scatter(xs, ys, color='red', s=10, alpha=0.2)
    plt.title('Distribution of Cars across the road')
    plt.axis('off')
    st.pyplot(plt)


def _show_points_distribution_birds_eye(coordinates_dataframe: pd.DataFrame, road_width: int):
    road_xs = [-road_width, road_width, road_width, -road_width, -road_width]
    road_ys = [0, 0, 500, 500, 0]
    plt.figure(figsize=(18, 18))
    plt.axes().set_aspect(1)
    plt.xlim(-50, 50)
    plt.ylim(0, 100)
    plt.fill(road_xs, road_ys, alpha=0.2, color='gray')
    plt.plot(
        [road_width / 2, road_width / 2], [0, 100],
        alpha=0.4, linewidth=4, color='white', ls='--'
    )
    plt.plot(
        [-road_width / 2, -road_width / 2], [0, 100],
        alpha=0.4, linewidth=4, color='white', ls='--'
    )
    plt.scatter(
        coordinates_dataframe['x'],
        np.sqrt(coordinates_dataframe['z'] ** 2 + coordinates_dataframe['y'] ** 2),
        color='red', s=10, alpha=0.1
    )
    plt.title('Distribution of Cars across the road from Bird\'s Eye perspective')
    plt.axis('off')
    st.pyplot(plt)


def _show_images_in_3d(dataframe: pd.DataFrame, n_samples: int, dataset_path: str):
    camera_matrix, _ = read_camera_intrinsic(dataset_path)
    image_ids = list(dataframe['ImageId'])
    prediction_strings = list(dataframe['PredictionString'])
    for index in range(n_samples):
        col1, col2 = st.beta_columns(2)
        image = Image.open(
            os.path.join(
                dataset_path,
                'train_images/{}.jpg'.format(image_ids[index])
            )
        )
        annotated_image = visualize_in_3d(
            np.array(image), string_to_coordinates(prediction_strings[index]),
            camera_matrix=camera_matrix
        )
        with col1:
            plt.figure(figsize=(18, 18))
            plt.imshow(image)
            plt.title(image_ids[index])
            plt.axis('off')
            st.pyplot(plt)
        with col2:
            plt.figure(figsize=(18, 18))
            plt.imshow(annotated_image)
            plt.title(image_ids[index] + '_annotated')
            plt.axis('off')
            st.pyplot(plt)


def explore_dataset(dataset_path: str):
    add_heading(
        content='Peking University/Baidu - Autonomous Driving Dataset',
        heading_level=1, align_center=True, add_hr=True
    )
    train_dataframe = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    coordinates_dataframe = get_coordinates_dataframe(dataframe=train_dataframe)
    with st.beta_expander(label='Train Dataframe', expanded=False):
        st.dataframe(data=train_dataframe)
    n_samples = st.sidebar.slider(
        'Please select number of samples to visualize',
        min_value=1, max_value=20, value=3
    )
    sampled_dataframe = train_dataframe.sample(n=n_samples)
    with st.beta_expander(
            label='Sample Train Images', expanded=True):
        _show_car_coordinates(
            copy.deepcopy(sampled_dataframe),
            n_samples=n_samples, dataset_path=dataset_path
        )
    with st.beta_expander(
            label='Distribution of Cars across the road', expanded=True):
        _show_points_distribution(
            sample_image_id=random.choice(list(train_dataframe['ImageId'])),
            dataframe=train_dataframe, dataset_path=dataset_path
        )
    with st.beta_expander(
            label='Distribution of Cars across the road from Bird\'s Eye perspective', expanded=True):
        _show_points_distribution_birds_eye(coordinates_dataframe=coordinates_dataframe, road_width=3)
    with st.beta_expander(
            label='Sample Train Images Annotated in 3D', expanded=True):
        _show_images_in_3d(
            dataframe=sampled_dataframe,
            n_samples=n_samples, dataset_path=dataset_path
        )
