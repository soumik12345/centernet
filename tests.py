import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from unittest import TestCase

from centernet.commons import read_camera_intrinsic
from centernet.dataloader import Preprocessor, PostProcess


class TestDataset(TestCase):
    
    def test_dataset(self):
        self.assertTrue(
            len(glob('./data/pku-autonomous-driving/*')) > 0)

    def test_camera_intrinsic(self):
        camera_intrinsic_file = './data/pku-autonomous-driving/camera/camera_intrinsic.txt'
        self.assertTrue(os.path.isfile(camera_intrinsic_file))
        (fx, fy, cx, cy), _, _ = read_camera_intrinsic('./data/pku-autonomous-driving/')
        self.assertTrue(fx == 2304.5479)
        self.assertTrue(fy == 2305.8757)
        self.assertTrue(cx == 1686.2379)
        self.assertTrue(cy == 1354.9849)
    
    def test_car_model_files(self):
        self.assertTrue(
            len(glob('./data/pku-autonomous-driving/car_models/*.pkl')) > 0)
        self.assertTrue(
            len(glob('./data/pku-autonomous-driving/car_models_json/*.json')) > 0)
    
    def test_images(self):
        n_train_images = len(glob('./data/pku-autonomous-driving/train_images/*.jpg'))
        self.assertTrue(n_train_images > 0)
    
    def test_csv(self):
        self.assertTrue(
            os.path.isfile('./data/pku-autonomous-driving/train.csv'))
        train_dataframe = pd.read_csv('./data/pku-autonomous-driving/train.csv')
        self.assertTrue(len(train_dataframe) > 0)
        self.assertTrue('ImageId' in train_dataframe)
        self.assertTrue('PredictionString' in train_dataframe)


class TestPreProcessor(TestCase):

    def test_shapes(self):
        train_dataframe = pd.read_csv('./data/pku-autonomous-driving/train.csv')
        preprocessor = Preprocessor(
            dataset_path='./data/pku-autonomous-driving/',
            image_height=320, image_width=1024, model_scale=8
        )
        for _ in range(50):
            sample_dataset = train_dataframe.sample(n=1)
            sample_image_id = list(sample_dataset['ImageId'])[0]
            sample_prediction_string = list(sample_dataset['PredictionString'])[0]
            sample_image = np.array(Image.open(
                './data/pku-autonomous-driving/train_images/{}.jpg'.format(sample_image_id)))
            preprocessed_image = preprocessor.preprocess_image(sample_image, flip=False)
            self.assertTrue(preprocessed_image.shape == (320, 1024, 3))
            preprocessed_image = preprocessor.preprocess_image(sample_image, flip=True)
            self.assertTrue(preprocessed_image.shape == (320, 1024, 3))
    
    def test_shape_label(self):
        train_dataframe = pd.read_csv('./data/pku-autonomous-driving/train.csv')
        preprocessor = Preprocessor(
            dataset_path='./data/pku-autonomous-driving/',
            image_height=320, image_width=1024, model_scale=8
        )
        for _ in range(50):
            sample_dataset = train_dataframe.sample(n=1)
            sample_image_id = list(sample_dataset['ImageId'])[0]
            sample_prediction_string = list(sample_dataset['PredictionString'])[0]
            sample_image = np.array(Image.open(
                './data/pku-autonomous-driving/train_images/{}.jpg'.format(sample_image_id)))
            mask, regression_target = preprocessor.get_targets(
                sample_image, sample_prediction_string, flip=False)
            self.assertTrue(mask.shape == (40, 128))
            self.assertTrue(regression_target.shape == (40, 128, 7))
            mask, regression_target = preprocessor.get_targets(
                sample_image, sample_prediction_string, flip=True)
            self.assertTrue(mask.shape == (40, 128))
            self.assertTrue(regression_target.shape == (40, 128, 7))
