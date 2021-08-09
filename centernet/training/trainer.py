import torch
import pandas as pd
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..model import UNetModel
from ..dataloader import PKUDataset
from .criterion import train_criterion


class Trainer:

    def __init__(
        self, image_height: int, image_width: int, model_scale: int) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.model_scale = model_scale
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.optimizer, self.learning_rate_scheduler = None, None, None
        self.train_dataloader, self.val_dataloader = None, None
    
    def build_dataloader(
        self, dataset_path: str, dataframe: pd.DataFrame,
        split_ratio: float, batch_size: int) -> None:
        train_dataframe, val_dataframe = train_test_split(
            dataframe, test_size=split_ratio, random_state=42)
        train_dataset = PKUDataset(
            train_dataframe, dataset_path, is_training=True,
            image_height=self.image_height,
            image_width=self.image_width,
            model_scale=self.model_scale
        )
        val_dataset = PKUDataset(
            val_dataframe, dataset_path, is_training=False,
            image_height=self.image_height,
            image_width=self.image_width,
            model_scale=self.model_scale
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
    
    def compile(
        self, efficientnet_alias: str = 'b0',
        pre_trained_backbone: bool = True,
        learning_rate: float = 1e-2) -> None:
        self.model = UNetModel(
            320, 1024, 8,
            efficientnet_alias,
            pre_trained_backbone
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters, lr=learning_rate)
    
    def _train_step(self, batch_index, images, masks, regression_targets):
        images = images.to(self.device)
        masks = masks.to(self.device)
        regression_targets = regression_targets.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = train_criterion(output, masks, regression_targets)
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return loss
    
    def training_episode(self):
        epoch_loss = 0
        for batch_index, data in enumerate(tqdm(self.train_dataloader)):
            images, masks, regression_targets = data
            epoch_loss += self._train_step(
                batch_index, images,
                masks, regression_targets
            ).data
        epoch_loss /= len(self.train_dataloader.dataset)
        return epoch_loss
    
    def evaluation_episode(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                images, masks, regression_targets = data
                images = images.to(self.device)
                masks = masks.to(self.device)
                regression_targets = regression_targets.to(self.device)
                output = self.model(images)
                epoch_loss += train_criterion(
                    output, masks,
                    regression_targets, size_average=False
                ).data
        epoch_loss /= len(self.val_dataloader.dataset)
        return epoch_loss
    
    def train_and_evaluate(self, epochs: int = 10):
        self.learning_rate_scheduler = lr_scheduler.StepLR(
            self.optimizer, gamma=0.1,
            step_size=max(epochs, 10) * len(self.train_dataloader) // 3
        )
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.training_episode()
            self.evaluation_episode()
