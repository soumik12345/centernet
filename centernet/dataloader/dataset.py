import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .preprocess import Preprocessor


class PKUDataset(Dataset):

    def __init__(
        self, dataframe: pd.DataFrame, dataset_path: str, is_training: bool,
        image_height: int, image_width: int, model_scale: int) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.is_training = is_training
        self.preprocessor = Preprocessor(
            image_height=image_height, image_width=image_width,
            model_scale=model_scale, dataset_path=dataset_path
        )
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        index = index.tolist() if torch.is_tensor(index) else index
        index, labels = self.dataframe.values[index]
        image_file = os.path.join(
            self.dataset_path,
            'train_images/{}.jpg'.format(index)
        )
        flip = np.random.randint(10) == 1 if self.is_training else False
        original_image = np.array(Image.open(image_file))
        image = self.preprocessor.preprocess_image(image=original_image, flip=flip)
        image = np.rollaxis(image, 2, 0)
        mask, regression_targets = self.preprocessor.get_targets(
            image=original_image, labels=labels, flip=flip
        )
        regression_targets = np.rollaxis(regression_targets, 2, 0)
        return image, mask, regression_targets
