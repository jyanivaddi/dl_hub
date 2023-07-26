import numpy as np
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch_lr_finder import LRFinder


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

