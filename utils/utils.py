import torchinfo
import numpy as np
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch_lr_finder import LRFinder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np



def compute_grad_cam_map(model, target_layers, input_tensor, targets=None , image_weight=0.2):
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grad_cam_output = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    grad_cam_output = grad_cam_output[0,:]
    un_norm_image = un_normalize_image(input_tensor.squeeze())
    un_norm_image_np = np.asarray(un_norm_image).transpose((1,2,0))
    grad_cam_map = show_cam_on_image(un_norm_image_np, grad_cam_output, use_rgb=True, image_weight = image_weight)
    return grad_cam_map


def get_incorrect_predictions(model, test_loader, device):
    model.eval()
    incorrect_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target,pred, output):
                if not p.eq(t.view_as(p)).item():
                    incorrect_predictions.append(
                        [d.cpu(), t.cpu(), p.cpu(),o[p.item()].cpu()]
                    )
    return incorrect_predictions


def generate_grad_cam_visualizations(model, target_layers, incorrect_predictions, class_names, num_images_to_compute=25, image_weight=0.2):
    grad_cam_map_list = []
    prediction_list = []
    ground_truth_list = []
    for cnt in range(num_images_to_compute):
        input_tensor = torch.unsqueeze(incorrect_predictions[cnt][0],0)
        ground_truth = class_names[str(incorrect_predictions[cnt][1].item())]
        prediction = class_names[str(incorrect_predictions[cnt][2].item())]
        prediction_list.append(prediction)
        ground_truth_list.append(ground_truth)
        grad_cam_map = compute_grad_cam_map(model, target_layers, input_tensor, image_weight=image_weight)
        grad_cam_map_list.append(grad_cam_map)
    return grad_cam_map_list, prediction_list, ground_truth_list


def find_best_lr(model, train_loader, optimizer, criterion, device):
    lr_finder = LRFinder(model, optimizer, criterion, device) 
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode='exp')
    lr_finder.plot()
    lr_finder.reset()
    return lr_finder.history


def un_normalize_image(img):
    un_norm_transform = transforms.Compose([transforms.Normalize((0.,0.,0.,),(1./0.247,1./0.244,1./0.262)),
                                                 transforms.Normalize((-0.491,-0.482,-0.447),(1.0,1.0,1.0))])
    return un_norm_transform(img)


def get_device():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def model_summary(model, input_size):
    torchinfo.summary(model, 
                      input_size = input_size, 
                      batch_dim=0, 
                      col_names=("kernel_size",
                                 "input_size",
                                 "output_size",
                                 "num_params",
                                 "mult_adds"),
                       verbose=1,) 


