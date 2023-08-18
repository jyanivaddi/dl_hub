import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pathlib import Path
import pytorch_lightning as pl
from .yolo_v3_utils.utils import check_class_accuracy, mean_average_precision, plot_couple_examples, get_evaluation_bboxes

def compute_grad_cam_map(self, inputs, index=0):
    
    output = self.net(inputs['image'])[0]
    output_nonmax = utils.non_max_suppression(output, conf_thres=0.25, iou_thres=0.45, multi_label=True)[0]
    print(output_nonmax)
    
    output_nonmax[:, :4] = utils.scale_coords(self.final_shape, output_nonmax[:, :4], self.ori_shape).round()

    
    scores = output_nonmax[:, 4]
    scores = scores.unsqueeze(0)
    print(scores.shape)
    score = torch.max(scores)
    #score = torch.min(scores)
    idx = scores.argmax().numpy()
    one_hot_output = torch.FloatTensor(1, scores.size()[-1]).zero_()
    one_hot_output[0][idx] = 1
    print(one_hot_output)
    print(score)

    self.net.zero_grad()

    scores.backward(gradient=one_hot_output, retain_graph = True)
    

    self.gradient = self.net.get_activations_gradient()

    self.feature = self.net.get_activations_features()

    print(self.gradient)
    #gradient_tensor = torch.tensor(np.array(self.gradient[2]))
    #pooled_gradients = torch.mean(gradient_tensor, dim=[0, 2, 3])
    
    target = self.feature[0].detach().numpy()[0]
    guided_gradients = self.gradient.detach().numpy()[0]
    weights = np.mean(guided_gradients, axis = (1, 2))  # take averages for each gradient
    print(weights.shape)
    print(target.shape)
    # create empty numpy array for cam
    cam = np.ones(target.shape[1:], dtype = np.float32)
    
    # multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights-1):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
    # comment this line if use colormap
    cam = 255 - cam
    # comment this line if use pixel matshow, otherwise cancel the comment
    #cam = np.uint8(cam * 255)  # scale between 0-255 to visualize
    
    
    # comment these two lines if use color map
    plt.matshow(cam.squeeze())
    plt.show()
    
    '''
    cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.ANTIALIAS))/255
    
    original_image = Image.open('./img/4.png')
    I_array = np.array(original_image)
    original_image = Image.fromarray(I_array.astype("uint8"))
    save_class_activation_images(original_image, cam, 'cam-featuremap')
    '''
    
    ################################## 
    # This is for pixel matplot method
    ##################################
    test_img = cv2.imread('./img/1.png')
    heatmap = cam.astype(np.float32)
    heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.6 + test_img
    cv2.imwrite('./new_map.jpg', superimposed_img)


    ################################################ 
    # Using these codes here, you can generate CAM map for each object 
    ################################################

    box = output_nonmax[idx][:4].detach().numpy().astype(np.int32)
    #print(box)
    x1, y1, x2, y2 = box
    ratio_x1 = x1 / test_img.shape[1]
    ratio_x2 = x2 / test_img.shape[0]
    ratio_y1 = y1 / test_img.shape[1]
    ratio_y2 = y2 / test_img.shape[0]

    x1_cam = int(cam.shape[1] * ratio_x1)
    x2_cam = int(cam.shape[0] * ratio_x2)
    y1_cam = int(cam.shape[1] * ratio_y1)
    y2_cam = int(cam.shape[0] * ratio_y2)

    cam = cam[y1_cam:y2_cam, x1_cam:x2_cam]
    cam = cv2.resize(cam, (x2 - x1, y2 - y1))


    class_id = output[idx][-1].detach().numpy()
    return cam, box, class_id
    

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