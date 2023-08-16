from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pathlib import Path
import pytorch_lightning as pl
from yolo_v3_utils.utils import check_class_accuracy, mean_average_precision


def calc_MAP(model, test_loader, config, scaled_anchors):
    plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str, every: int = 1, verbose:bool = False):
        super().__init__()
        self.every = every
        self.dirpath = dirpath
        self.verbose=verbose

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if self.every >=1 and (trainer.current_epoch +1) % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"checkpoint_epoch_{trainer.current_epoch}_step_{pl_module.global_step}.ckpt"
            trainer.save_checkpoint(current)
  
    
            
def train_yolov3_model(model, datamodule, ckpt_path=None, epochs = 2):
    trainer = Trainer(
        enable_checkpointing=True,
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), 
                   TQDMProgressBar(refresh_rate=10), 
                   PeriodicCheckpoint(dirpath="logs/",every=5, verbose=True)],
        num_sanity_val_steps=0,
        precision=16
    )
    
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(),ckpt_path=ckpt_path)
    trainer.test(model, datamodule.test_dataloader())
    return trainer