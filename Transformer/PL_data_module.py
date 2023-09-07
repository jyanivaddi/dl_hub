import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
# Huggingface datasets and tokenizers
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from .dataset import BilingualDataset
#from dataset import BilingualDataset


class OpusDataModule(LightningDataModule):
    def __init__(self,
                 config,
                 val_split=0.1,
                 num_workers=os.cpu_count()-1
                 ):

        # Initialize the class. Set up the datadir, image dims, and num classes
        super().__init__()
        self.config = config
        self.val_split = val_split
        self.num_workers = num_workers
        self.ds_pruned = None
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
            ds_raw = load_from_disk(self.config['ds_path'])
        else:
            ds_raw = load_dataset(self.config['ds_name'],
                                  f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                                  split='train')
        if self.config['save_ds_to_disk']:
            ds_raw.save_to_disk(f"./{self.config['ds_name']}_{self.config['lang_src']}_{self.config['lang_tgt']}")

        # Build tokenizers
        self.tokenizer_src = self._get_or_build_tokenizer(ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = self._get_or_build_tokenizer(ds_raw, self.config['lang_tgt'])

        # Prune dataset
        self.ds_pruned = self.prune_dataset(ds_raw)


    def prepare_data(self) -> None:
        self.get_dataset()


    def prune_dataset(self, train_ds_raw):
        """
        Trim the dataset so that the longest sentence is more than 120
        """
        lang_src = self.config["lang_src"]
        lang_tgt = self.config["lang_tgt"]
        sorted_train_ds = sorted(train_ds_raw, key=lambda x: (len(x["translation"][lang_src])))
        #filtered_sorted_train_ds = [k for k in sorted_train_ds if len(k["translation"][lang_src]) < 120]
        #filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(k["translation"][lang_tgt]) < 120]
        #filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(k["translation"][lang_src]) + 10 > len(k["translation"][lang_tgt])]
        filtered_sorted_train_ds = [k for k in sorted_train_ds if len(k["translation"][lang_src]) + 10 > len(k["translation"][lang_tgt])]

        # transform the text into tokens
        filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(self.tokenizer_src.encode(k["translation"][lang_src]).ids) < 150]
        return filtered_sorted_train_ds

    def setup(self, stage=None):
        # keep 90% for training, 10% for validation
        train_proportion = max(1. - self.val_split, 0.9)
        train_ds_size = int(train_proportion * len(self.ds_pruned))
        val_ds_size = len(self.ds_pruned) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(self.ds_pruned, [train_ds_size, val_ds_size])

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
                                       collate_fn = self.train_dataset.collate_samples,
                                       #collate_fn = None,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=True)
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(self.train_eval_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=self.train_eval_dataset.collate_samples,
                                     num_workers=self.num_workers,
                                     pin_memory=True)

        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(self.train_eval_dataset,
                                      batch_size=1,
                                      shuffle=True,
                                      collate_fn=self.train_eval_dataset.collate_samples,
                                      num_workers=self.num_workers,
                                      pin_memory=True)
        return test_data_loader

        