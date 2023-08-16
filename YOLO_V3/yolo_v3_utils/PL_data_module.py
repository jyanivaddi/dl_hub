import os
import torch
import numpy as np
from typing import List, Any
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, seed_everything
from dl_hub.YOLO_V3.yolo_v3_utils.pascal_voc_dataset import YOLODataset
from dl_hub.YOLO_V3.yolo_v3_utils.pascal_voc_dataset_mosaic import YOLODatasetMosaicAugmentation


class YOLODataModule(LightningDataModule):
    def __init__(self,
                 csv_files,
                 img_dir,
                 label_dir,
                 anchors,
                 batch_size,
                 image_size=416,
                 S=[13, 26, 52],
                 C=20,
                 train_transforms = None,
                 val_transforms = None,
                 test_transforms = None,
                 val_split=0.2,
                 num_workers = 1,
                 use_mosaic_on_train = True,
                 mosaic_probability=0.75,
                 pin_memory = False):

        # Initialize the class. Set up the datadir, image dims, and num classes
        super().__init__()
        self.train_csv_path, self.test_csv_path = csv_files
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = anchors
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.val_split = val_split
        self.S = S
        self.C = C
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_dataset = None
        self.train_dataset = None
        self.train_eval_dataset = None
        self.mosaic_probability=mosaic_probability
        self.use_mosaic_on_train = use_mosaic_on_train

    def get_dataset_train(self):
        if self.use_mosaic_on_train:
            train_dataset = YOLODatasetMosaicAugmentation(self.train_csv_path,
                                                          self.img_dir,
                                                          self.label_dir,
                                                          self.anchors,
                                                          image_size=self.image_size,
                                                          S=self.S,
                                                          C=self.C,
                                                          mosaic_probability=self.mosaic_probability,
                                                          transform=self.train_transforms)
        else:
            train_dataset = YOLODataset(self.train_csv_path,
                                              self.img_dir,
                                              self.label_dir,
                                              self.anchors,
                                              image_size=self.image_size,
                                              S=self.S,
                                              C=self.C,
                                              transform=self.train_transforms)
        return train_dataset



    def get_dataset_test(self):
        return YOLODataset(self.test_csv_path,
                           self.img_dir,
                           self.label_dir,
                           self.anchors,
                           image_size=self.image_size,
                           S=self.S,
                           C=self.C,
                           transform=self.test_transforms)

    def get_dataset_val(self):
        return YOLODataset(self.train_csv_path,
                           self.img_dir,
                           self.label_dir,
                           self.anchors,
                           image_size=self.image_size,
                           S=self.S,
                           C=self.C,
                           transform=self.test_transforms)


    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits


    def prepare_data(self):
        # Download the dataset
        YOLODataset(self.train_csv_path,
                    self.img_dir,
                    self.label_dir,
                    self.anchors,
                    image_size=self.image_size,
                    S=self.S,
                    C=self.C,
                    transform=self.train_transforms)
        YOLODataset(self.test_csv_path,
                    self.img_dir,
                    self.label_dir,
                    self.anchors,
                    image_size=self.image_size,
                    S=self.S,
                    C=self.C,
                    transform=self.test_transforms)
        return

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            dataset_train = self.get_dataset_train()
            dataset_val = self.get_dataset_val()

            # Split
            self.train_dataset = self._split_dataset(dataset_train)
            self.train_eval_dataset = self._split_dataset(dataset_val, train=False)

        if stage == 'test' or stage:
            self.test_dataset = self.get_dataset_test()

    def train_dataloader(self):
        train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False)
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(
            dataset=self.train_eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False)
        return val_data_loader

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = self.get_dataset_test()
        test_data_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False)
        return test_data_loader

        