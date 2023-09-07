import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback
import torchmetrics
from .config import get_weights_file_path
#from config import get_weights_file_path
import pytorch_lightning as pl


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch}")
        trainer.save_checkpoint(model_filename)


class PrintAccuracyAndLoss(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        trainer.model.log("train_epoch_loss", train_loss)
        print(f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        assert len(trainer.model.predicted_list) > 0, "Validation: predicted list is empty"
        assert len(trainer.model.expected_list) > 0, "Validation: expected list is empty"

        # compute the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(trainer.model.predicted_list, trainer.model.expected_list)

        # compute word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(trainer.model.predicted_list, trainer.model.expected_list)

        # compute the BLEU metric
        metric = torchmetrics.BLEUScore(n_gram=2)
        bleu = metric(trainer.model.predicted_list, trainer.model.expected_list)

        trainer.model.log("validation_epoch_wer", wer)
        trainer.model.log("validation_epoch_cer", cer)
        trainer.model.log("validation_epoch_bleu", bleu)
        trainer.model.predicted_list = []
        trainer.model.expected_list = []
        assert len(trainer.model.predicted_list) == 0, "Validation: predicted list is not reset"
        assert len(trainer.model.expected_list) == 0, "Validation: expected list is not reset"
        return

def train_transformer(model, datamodule, config, ckpt_path=None, epochs=2):
    trainer = Trainer(
        enable_checkpointing=True,
        max_epochs=epochs,
        accelerator="auto",
        #accelerator=None,
        devices=1 if torch.cuda.is_available() else None,
        #logger=CSVLogger(save_dir="logs/"),
        logger=TensorBoardLogger(save_dir=config["rundir"], name=config["experiment_name"], default_hp_metric=False),
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10),
                   #RichProgressBar(refresh_rate=10, leave=True),
                   PeriodicCheckpoint(config, verbose=True),
                   PrintAccuracyAndLoss()],
        num_sanity_val_steps=0,
        precision=16
    )
    
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt_path)
    trainer.test(model, datamodule.test_dataloader())
    return trainer