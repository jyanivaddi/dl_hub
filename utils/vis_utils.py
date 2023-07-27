import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from prettytable import PrettyTable
from utils.utils import un_normalize_image

def print_train_log(train_acc, test_acc, train_loss, test_loss):
    t = PrettyTable()
    t.field_names=["Epoch", "Train loss", "Val loss", "Train Accuracy", "Val Accuracy"]
    for cnt in range(len(train_acc)):
        t.add_row([cnt+1,train_loss[cnt], test_loss[cnt], train_acc[cnt], test_acc[cnt]])
    print(t)


def preview_images(train_loader, class_names, num_rows = 5, num_cols = 5):
    batch_data, batch_label = next(iter(train_loader))
    num_images_to_preview = num_rows*num_cols
    inv_transforms = transforms.Compose([transforms.Normalize((0.,0.,0.,),
                                            (1./0.247,1./0.244,1./0.262)),
                                        transforms.Normalize((-0.491,-0.482,-0.447),
                                                             (1.0,1.0,1.0))])
    for cnt in range(num_images_to_preview):
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.tight_layout()
        normalized_tensor_img = inv_transforms(batch_data[cnt].squeeze())
        this_img = np.asarray(normalized_tensor_img)
        this_img = (this_img*255./np.max(this_img)).astype('uint8')
        plt.imshow(this_img.transpose((1,2,0)))
        plt.title(class_names[str(batch_label[cnt].item())])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_image_grid(images_list, prediction_list, ground_truth_list, num_rows, num_cols):
    num_images_to_preview = num_rows*num_cols
    fig = plt.figure()
    for cnt, this_img in enumerate(images_list):
        ax = fig.add_subplot(num_rows, num_cols, cnt+1,xticks=[],yticks=[])
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.imshow(this_img.transpose((1,2,0)))
        title_str = f"{prediction_list[cnt]}/{ground_truth_list[cnt]}"
        ax.set_title(title_str,fontsize=8)
        if cnt == num_images_to_preview:
            break
    plt.tight_layout()
    plt.show()


def show_incorrect_predictions(incorrect_predictions, class_names, num_rows = 5, num_cols = 5):
    incorrect_predictions_numpy_list = []
    num_images_to_show = num_rows*num_cols
    prediction_list = []
    ground_truth_list = []
    for this_pred in incorrect_predictions[:num_images_to_show]:
        orig_img = this_pred[0]
        inv_transforms = transforms.Compose([transforms.Normalize((0.,0.,0.,),
                                            (1./0.247,1./0.244,1./0.262)),
                                        transforms.Normalize((-0.491,-0.482,-0.447),
                                                             (1.0,1.0,1.0))])
        ground_truth = class_names[str(this_pred[1])]
        ground_truth_list.append(ground_truth)
        prediction = class_names[str(this_pred[2])]
        prediction_list.append(prediction)
        un_normalized_tensor_img = inv_transforms(orig_img.squeeze())
        un_normalized_numpy_img = np.asarray(un_normalized_tensor_img)
        incorrect_predictions_numpy_list.append(un_normalized_numpy_img.transpose((1,2,0)))
    # show incorrect predictions
    plot_image_grid(incorrect_predictions_numpy_list, prediction_list, ground_truth_list, num_rows, num_cols)


def plot_statistics(train_losses, train_acc, test_losses, test_acc, target_test_acc = 99):
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].plot(test_losses)
    axs[0, 0].legend(('Train','Test'),loc='best')
    #axs[0, 0].set_title("Loss")
    axs[0, 1].plot(train_acc)
    axs[0, 1].plot(test_acc)
    axs[0, 1].axhline(target_test_acc, color='r')
    axs[0, 0].legend(('Train','Test'),loc='best')
    axs[1, 0].set_title("Accuracy")


def preview_augmentations(train_loader, image_transform):
    batch_data, batch_label = next(iter(train_loader))
    preview_img = np.asarray(un_normalize_image(batch_data[0].squeeze()))
    preview_img = (preview_img*255./np.max(preview_img)).astype('uint8')
    preview_label = batch_label[0]
    fig, axs = plt.subplots(1, 2)
    plt.tight_layout()

    un_normalized_img = un_normalize_image(batch_data[0].squeeze())
    un_normalized_img = np.asarray(un_normalized_img)
    transformed_numpy_img = image_transform(image = un_normalized_img.transpose(1,2,0))["image"]
    axs[0].imshow(preview_img.transpose((1,2,0)))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(transformed_numpy_img)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_lr_values(lr_list):
    num_epochs = len(lr_list)
    plt.figure()
    plt.plot(range(1,num_epochs+1),lr_list)
    plt.xlabel('Epoch #')
    plt.ylabel("Learning Rate")
    plt.show()


def plot_lr_values_one_cycle_policy(scheduler, num_epochs, num_batches):
    lrs = []
    steps = []
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
            steps.append(epoch * num_batches + batch)

    plt.figure()
    plt.legend()
    plt.plot(steps, lrs, label='Learning Rate')
    plt.xlabel('Step #')
    plt.ylabel("Learning Rate")
    plt.show()


def plot_losses(train_losses, test_losses):
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    epochs = range(1,len(train_losses)+1)
    axs[0].plot(epochs, train_losses)
    axs[0].set_title("Train")
    axs[1].plot(epochs, test_losses)
    axs[1].set_title("Test")


def plot_accuracy(train_acc, test_acc, target_test_acc = 90.):
    epochs = range(1,len(train_acc)+1)
    plt.figure()
    plt.plot(epochs, train_acc, epochs, test_acc)
    plt.axhline(target_test_acc, color='r')
    plt.legend(('Train','Test'),loc='best')
    plt.title("Accuracy")