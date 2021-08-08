import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from random import sample
from scipy.sparse.construct import random
import streamlit as st

from .commons import plot_image_matplotlib
from ..dataloader import Preprocessor, PostProcess, PKUDataset
from ..commons import get_coordinates_dataframe, visualize_in_3d


def _show_pre_process(dataset_path: str, train_dataframe: pd.DataFrame):
    n_samples = st.sidebar.slider(
        'Please select number of preprocessor demos',
        min_value=1, max_value=20, value=3
    )
    preprocessor = Preprocessor(
        dataset_path=dataset_path, image_height=320,
        image_width=1024, model_scale=8
    )
    sample_dataset = train_dataframe.sample(n=n_samples)
    sample_image_ids = list(sample_dataset['ImageId'])
    sample_prediction_strings = list(sample_dataset['PredictionString'])
    for index in range(n_samples):
        sample_image_id = sample_image_ids[index]
        sample_image = np.array(Image.open(
            os.path.join(dataset_path, 'train_images/{}.jpg'.format(sample_image_id))
        ))
        sample_mask, sample_regression_target = preprocessor.get_targets(
            sample_image, sample_prediction_strings[index], flip=False
        )
        with st.expander(
                label='Sample Train Images and Labels: {} '.format(sample_image_id), expanded=True):
            plot_image_matplotlib(
                image=preprocessor.preprocess_image(
                    image=sample_image, flip=False), title=sample_image_id)
            plot_image_matplotlib(
                image=sample_mask, title=sample_image_id + '_mask')
            plot_image_matplotlib(
                image=sample_regression_target[:, :, -2],
                title=sample_image_id + '_regression_targets')


def _show_post_process(
        dataset_path: str, train_dataframe: pd.DataFrame,
        coordinate_dataframe: pd.DataFrame, n_samples: int):
    sample_image_shape = np.array(Image.open(
        glob(os.path.join(dataset_path, 'train_images/*'))[0])).shape
    sample_dataset = train_dataframe.sample(n=n_samples)
    sample_image_ids = list(sample_dataset['ImageId'])
    sample_prediction_strings = list(sample_dataset['PredictionString'])
    for index in range(n_samples):
        sample_image = np.array(Image.open(os.path.join(
            dataset_path, 'train_images/{}.jpg'.format(sample_image_ids[index]))))
        preprocessor = Preprocessor(
            dataset_path=dataset_path, image_height=320,
            image_width=1024, model_scale=8
        )
        post_processor = PostProcess(
            image_height=320, image_width=1024,
            model_scale=8, distance_threshold=2,
            image_shape=sample_image_shape, dataset_path=dataset_path
        )
        post_processor.fit_coordinates(coordinates_dataframe=coordinate_dataframe)
        sample_mask, sample_regression_target = preprocessor.get_targets(
            sample_image, sample_prediction_strings[index], flip=False
        )
        sample_regression_target = np.rollaxis(sample_regression_target, 2, 0)
        coordinates = post_processor.extract_coordinates(
            np.concatenate([sample_mask[None], sample_regression_target], 0), flipped=False)
        plot_image_matplotlib(image=visualize_in_3d(
            image=sample_image, coordinates=coordinates,
            camera_matrix=post_processor.camera_matrix
        ), title=sample_image_ids[index])


def _show_dataset_demo(dataset_path: str, dataframe: pd.DataFrame, n_samples: int):
    dataset = PKUDataset(
        dataframe=dataframe, dataset_path=dataset_path, is_training=True,
        image_height=320, image_width=1024, model_scale=8
    )
    st.markdown('**Dataset Length: {}**'.format(len(dataset)))
    indices = sample(list(range(len(dataset))), n_samples)
    for index in indices:
        image, mask, regression_target = dataset[index]
        plot_image_matplotlib(
            np.rollaxis(image, 0, 3), title=str(index))
        plot_image_matplotlib(
            mask, title='{}_mask'.format(index))
        plot_image_matplotlib(
            regression_target[-2], title='{}_regression_target'.format(index))


def data_loader_module(dataset_path: str):
    train_dataframe = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    coordinates_dataframe = get_coordinates_dataframe(dataframe=train_dataframe)
    _show_pre_process(dataset_path=dataset_path, train_dataframe=train_dataframe)
    n_samples = st.sidebar.slider(
        'Please select number of postprocessor demos',
        min_value=1, max_value=20, value=3
    )
    with st.expander(label='Postprocessor Demos', expanded=True):
        _show_post_process(
            dataset_path=dataset_path,
            train_dataframe=train_dataframe,
            coordinate_dataframe=coordinates_dataframe,
            n_samples=n_samples
        )
    n_samples = st.sidebar.slider(
        'Please select number of dataset demos',
        min_value=1, max_value=20, value=3
    )
    with st.expander(label='Dataset Demos', expanded=True):
        _show_dataset_demo(
            dataset_path=dataset_path,
            dataframe=train_dataframe,
            n_samples=n_samples
        )
