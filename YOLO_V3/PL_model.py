import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy


class LitYOLOv3(LightningModule):
    def __init__(self,
                 loss_criterion,
                 scaled_anchors,
                 conf_threshold,
                 optimizer=None,
                 scheduler_dict=None,
                 num_classes=10,
                 epochs=20):
        super().__init__()

        self.save_hyperparameters()
        self.model = YOLOv3(num_classes=num_classes)
        self.loss_criterion = loss_criterion
        self.scaled_anchors = scaled_anchors
        self.conf_threshold = conf_threshold
        self.optimizer = None
        self.scheduler_dict = None
        self.epochs = epochs

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler_dict(self, scheduler, freq='step'):
        self.scheduler = scheduler
        self.scheduler_dict = {
            "scheduler": self.scheduler,
            "interval": freq,
        }

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (y[0],y[1],y[2])

        out = self(x)
        loss = (
                self.loss_criterion(out[0], y0, self.scaled_anchors[0]) +
                self.loss_criterion(out[1], y1, self.scaled_anchors[1]) +
                self.loss_criterion(out[2], y2, self.scaled_anchors[2])
            )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        """
        Evaluate the model on validation dataset.
        we compute the class accuracy, the no object accuracy, and the object accuracy
        """

        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        x, y = batch
        out = self(x)

        for i in range(3):
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > self.conf_threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        if stage:
            class_acc = (correct_class/(tot_class_preds+1e-16))*100
            no_obj_acc = (correct_noobj/(tot_noobj+1e-16))*100
            obj_acc = (correct_obj/(tot_obj+1e-16))*100
            self.log(f"{stage}_Class_Accuracy",class_acc, prog_bar=True)
            self.log(f"{stage}_No_Obj_Accuracy", no_obj_acc, prog_bar=True)
            self.log(f"{stage}_Obj_Accuracy",obj_acc, prog_bar=True)
            print(f"epoch: {self.trainer.current_epoch}  {stage}_Class_Accuracy: {class_acc}")
            print(f"epoch: {self.trainer.current_epoch}  {stage}_No_Obj_Accuracy: {no_obj_acc}")
            print(f"epoch: {self.trainer.current_epoch}  {stage}_Obj_Accuracy: {obj_acc}")


    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_id):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_dict}


