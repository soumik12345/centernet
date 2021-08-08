import os
import pandas as pd
from glob import glob
from unittest import TestCase


class TestDataset(TestCase):
    
    def test_dataset(self):
        self.assertTrue(
            len(glob('./data/pku-autonomous-driving/*')) > 0)

    def test_camera_intrinsic(self):
        self.assertTrue(
            os.path.isfile('./data/pku-autonomous-driving/camera/camera_intrinsic.txt'))
    
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
