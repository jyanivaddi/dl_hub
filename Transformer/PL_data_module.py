from pathlib import Path
import torch
from typing import List, Any
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
# Huggingface datasets and tokenizers
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from .dataset import BilingualDataset


class OpusDataModule(LightningDataModule):
    def __init__(self,
                 config,
                 val_split=0.1
                 ):

        # Initialize the class. Set up the datadir, image dims, and num classes
        super().__init__()
        self.config = config
        self.val_split = val_split
        self.ds_raw = None
        self.tokenizer_src = None
        self.tokenizer_tgt = None
        self.train_dataset = None
        self.train_eval_dataset = None
        self.test_dataset = None


    @staticmethod
    def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]

    def _get_or_build_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            # code inspired from huggingface tokenizers
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                       min_frequency=2)
            tokenizer.train_from_iterator(OpusDataModule.get_all_sentences(ds, lang),
                                          trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def get_dataset(self):
        if self.config['ds_mode'] == "disk":
            self.ds_raw = load_from_disk(self.config['ds_path'])
        else:
            self.ds_raw = load_dataset(self.config['ds_name'],
                                  f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                                  split='train')
        if self.config['save_ds_to_disk']:
            self.ds_raw.save_to_disk(f"./{self.config['ds_name']}")

        # Build tokenizers
        self.tokenizer_src = self._get_or_build_tokenizer(self.ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = self._get_or_build_tokenizer(self.ds_raw, self.config['lang_tgt'])

    def prepare_data(self) -> None:
        self.get_dataset()

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

    def setup(self, stage=None):
        train_ds_raw = self._split_dataset(self.ds_raw)
        val_ds_raw = self._split_dataset(self.ds_raw, train=False)

        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            # Split

            self.train_dataset = BilingualDataset(train_ds_raw,
                                                  self.tokenizer_src,
                                                  self.tokenizer_tgt,
                                                  self.config['lang_src'],
                                                  self.config['lang_tgt'],
                                                  self.config['seq_len'])
            self.train_eval_dataset = BilingualDataset(val_ds_raw,
                                                       self.tokenizer_src,
                                                       self.tokenizer_tgt,
                                                       self.config['lang_src'],
                                                       self.config['lang_tgt'],
                                                       self.config['seq_len'])

        # Assign test dataset
        if stage == 'test' or stage:
            self.test_dataset = BilingualDataset(val_ds_raw,
                                                 self.tokenizer_src,
                                                 self.tokenizer_tgt,
                                                 self.config['lang_src'],
                                                 self.config['lang_tgt'],
                                                 self.config['seq_len'])

    def train_dataloader(self):

        train_data_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.config['batch_size'],
                                       shuffle=True,
                                       num_workers = 2)
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(self.train_eval_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers = 2 )
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(self.train_eval_dataset,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers = 2)
        return test_data_loader

        