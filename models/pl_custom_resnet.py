import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from .custom_resnet import CustomResnet


class LitResnet(LightningModule):
    def __init__(self, loss_criterion, lr=0.01, drop_out_probability=0.02, num_classes=10,
                 base_channels=3, epochs=20):
        super().__init__()

        self.save_hyperparameters()
        self.model = CustomResnet(base_channels=base_channels,
                                  num_classes=num_classes,
                                  drop_out_probability=drop_out_probability)
        self.loss_criterion = loss_criterion
        self.optimizer = None
        self.scheduler_dict = None
        self.epochs = epochs

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler_dict(self, scheduler_dict):
        self.scheduler_dict = scheduler_dict

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        train_acc = accuracy(preds, y,"multiclass", num_classes=10)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        #loss = nn.CrossEntropyLoss(logits, y, reduction='mean')
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y,"multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_id):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_dict}


