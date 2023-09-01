import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar, RichProgressBar
from pathlib import Path
from .config import get_weights_file_path
import pytorch_lightning as pl


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch}")
        torch.save({
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.model.optimizer.state_dict(),
            'global_step': trainer.global_step
        }, model_filename)


def train_transformer(model, datamodule, config, ckpt_path=None, epochs=2):
    trainer = Trainer(
        enable_checkpointing=True,
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10),
                   #RichProgressBar(refresh_rate=10, leave=True),
                   PeriodicCheckpoint(config, verbose=True)],
        num_sanity_val_steps=0,
        precision=32
    )
    
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt_path)
    trainer.test(model, datamodule.test_dataloader())
    return trainer