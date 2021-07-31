import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt

from ..dataloader import Preprocessor


def data_loader_module(dataset_path: str):
    train_dataframe = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    preprocessor = Preprocessor(
        dataset_path=dataset_path, image_height=320,
        image_width=1024, model_scale=8
    )
    n_samples = st.sidebar.slider(
        'Please select number of samples to visualize',
        min_value=1, max_value=20, value=3
    )
    sample_dataset = train_dataframe.sample(n=n_samples)
    sample_image_ids = list(sample_dataset['ImageId'])
    sample_prediction_strings = list(sample_dataset['PredictionString'])
    for index in range(n_samples):
        with st.beta_expander(
                label='Sample Train Images and Labels', expanded=True):
            sample_image_id = sample_image_ids[index]
            sample_image = np.array(Image.open(
                os.path.join(dataset_path, 'train_images/{}.jpg'.format(sample_image_id))
            ))
            sample_mask, sample_regression_target = preprocessor.get_targets(
                sample_image, sample_prediction_strings[index], flip=False
            )
            plt.figure(figsize=(18, 18))
            plt.imshow(preprocessor.preprocess_image(image=sample_image, flip=False))
            plt.title(sample_image_id)
            plt.axis('off')
            st.pyplot(plt)
            plt.figure(figsize=(18, 18))
            plt.imshow(sample_mask)
            plt.title(sample_image_id + '_mask')
            plt.axis('off')
            st.pyplot(plt)
            plt.figure(figsize=(18, 18))
            plt.imshow(sample_regression_target[:, :, -2])
            plt.title(sample_image_id + '_regression_targets')
            plt.axis('off')
            st.pyplot(plt)
            st.markdown('<hr>', unsafe_allow_html=True)
