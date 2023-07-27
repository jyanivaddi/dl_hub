import torch
import torch.optim as optim
import torch.nn.functional as F
import models.resnet 
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from utils.utils import get_device
from dataloaders.data_loader import load_dataset


def validate_the_model(model, device, test_loader, loss_func, test_acc, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.*correct/len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_losses.append(test_loss)
    return 


def train_the_model(params, model, device, train_loader, optimizer, scheduler, loss_func, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if scheduler and params['scheduler_type'].upper() == 'ONECYCLELR':
            scheduler.step()
        correct+= output.argmax(dim=1).eq(target).sum().item()
        processed+= len(data)
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return  


def define_model(params):
    model_type = params['model_type']
    if model_type.upper() == 'RESNET18':
        model = models.resnet.ResNet18()
    return model


def define_optimizer(params, model):
    optimizer_type = params['optimizer_type']
    optimizer_params = params['optimizer_params']
    if optimizer_type.upper() == 'ADAM':
        if 'weight_decay' in optimizer_params:
            weight_decay = optimizer_params['weight_decay']
        else:
            weight_decay =  1e-4
        optimizer = optim.Adam(model.parameters(), 
                               lr = optimizer_params['lr'], 
                               weight_decay=weight_decay)
    elif optimizer_type.upper() == 'SGD':
        if 'weight_decay' in optimizer_params:
            weight_decay = optimizer_params['weight_decay']
        else:
            weight_decay =  5e-4
        if 'momentum' in optimizer_params:
            momentum = optimizer_params['momentum']
        else:
            momentum = 0.9
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_params['lr'],
                              momentum=momentum,
                              weight_decay=weight_decay)
    return optimizer


def define_scheduler(params, optimizer):
    scheduler_type = params['scheduler_type']
    scheduler_params = params['scheduler_params']
    if scheduler_type.upper() == 'ONECYCLELR':
        scheduler = OneCycleLR(
            optimizer, 
            max_lr = scheduler_params['max_lr'], 
            steps_per_epoch = scheduler_params['num_steps_per_epoch'],
            epochs = params['num_epochs'], 
            pct_start = scheduler_params['pct_start'],
            div_factor = scheduler_params['div_factor'], 
            three_phase = scheduler_params['three_phase'], 
            final_div_factor = scheduler_params['final_div_factor'], 
            anneal_strategy = scheduler_params['anneal_strategy'], 
            verbose=True)
    elif scheduler_type.upper() == 'REDUCELRONPLATEAU':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_params['mode'], 
            factor=scheduler_params['factor'], 
            patience=scheduler_params['patience'], 
            threshold=scheduler_params['threshold'],
            verbose=True)
    else:
        scheduler = None
    return scheduler


def get_dataset(params):
    dataset_name = params['dataset_name']
    train_transforms = params['train_transforms']
    test_transforms = params['test_transforms']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader, test_loader, class_names = load_dataset(dataset_name, train_transforms, test_transforms, batch_size, **kwargs)
    return train_loader, test_loader, class_names


def build_model(model, device, train_loader, test_loader, optimizer, scheduler, params):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    lr_values = []
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']

    for epoch in range(1,num_epochs+1):
        lr_values.append(optimizer.param_groups[0]['lr'])
        print(f"epoch: {epoch}\t learning rate: {scheduler.get_last_lr()[0]}")
        train_the_model(model, device, train_loader, optimizer, scheduler, loss_func, train_acc, train_losses)
        validate_the_model(model, device, test_loader, loss_func, test_acc, test_losses)
        if scheduler:
            scheduler.step()
    return train_losses, test_losses, train_acc, test_acc


def setup_model(params):
    device = get_device()
    torch.manual_seed(1)
    train_loader, test_loader, class_names = get_dataset(params)
    model = define_model(params)
    optimizer = define_optimizer(params, model)
    scheduler = define_scheduler(params, optimizer)
    return device, train_loader, test_loader, class_names, model, optimizer, scheduler
