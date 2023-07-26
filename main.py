import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import torch


def model_test(model, device, test_loader, loss_func, test_acc, test_losses):
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


def training_loop(model, device, train_loader, optimizer, scheduler, loss_func, train_acc, train_losses):
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
        scheduler.step()
        correct+= output.argmax(dim=1).eq(target).sum().item()
        processed+= len(data)
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return  

def setup_model():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    resnet_model = CustomResnet(base_channels=3,num_classes=10).to(device)
    model_summary(resnet_model, input_size=(3,32,32))

    train_transforms = A.Compose(
        [
            AA.crops.transforms.RandomResizedCrop(height = 32,width = 32,p=0.2),
            A.HorizontalFlip(p=0.2),
            AA.dropout.coarse_dropout.CoarseDropout(max_holes = 1, max_height=8,
                                                    max_width=8, min_holes = 1,
                                                    min_height=8, min_width=8,
                                                    fill_value=(0.491, 0.482, 0.447),
                                                    mask_fill_value = None),
    
            A.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262)),
            ToTensorV2(),
        ]
    )
    test_transforms = A.Compose([
    
        A.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262)),
        ToTensorV2(),
    ])

    torch.manual_seed(1)
    batch_size = 512
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader, class_names = load_cifar10_data(train_transforms, test_transforms, batch_size, **kwargs)

    # Reload train and test loader to preview augmentations
    torch.manual_seed(1)
    batch_size = 512
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    eg_train_loader, eg_test_loader, eg_class_names = load_cifar10_data(train_transforms= A.Compose([A.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262)),ToTensorV2()]), test_transforms=A.Compose([A.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262)),ToTensorV2()]), batch_size=32, **kwargs)

    # Random Resized Crop
    img_transforms = A.Compose([AA.crops.transforms.RandomResizedCrop(height= 32,width = 32,p=0.2)])
    preview_augmentations(eg_train_loader, img_transforms)

    # Horizontal Flip
    img_transforms = A.Compose([A.HorizontalFlip(always_apply=True)])
    preview_augmentations(eg_train_loader, img_transforms)

    # Cut out
    img_transforms = A.Compose([AA.dropout.coarse_dropout.CoarseDropout(max_holes = 1, max_height=8,
                                                    max_width=8, min_holes = 1,
                                                    min_height=8, min_width=8,
                                                    fill_value=(0.491, 0.482, 0.447),
                                                    mask_fill_value = None, always_apply=True)])
    preview_augmentations(eg_train_loader, img_transforms)

    preview_images(train_loader,class_names, num_rows = 5, num_cols = 5)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    lr_values = []
    max_lr = 4.65e-2
    # Define scheduler
    optim_obj.define_scheduler(max_lr)

    for epoch in range(1,num_epochs+1):
        lr_values.append(optim_obj.scheduler.get_lr())
        print(f"epoch: {epoch}\t learning rate: {optim_obj.scheduler.get_last_lr()[0]}")
        this_train_loss = training_loop(resnet_model, device, train_loader, optim_obj.optimizer, optim_obj.scheduler, criterion, train_acc, train_losses)
        this_loss = model_test(resnet_model, device, test_loader, criterion, test_acc, test_losses)
        #optim_obj.scheduler.step()
    plot_lr_values(this_scheduler, num_epochs, len(train_loader))
    plot_losses(train_losses, test_losses)
    plot_accuracy(train_acc, test_acc, target_test_acc=90.)
    print_train_log(train_acc, test_acc, train_losses, test_losses)
j
