import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms

from datasets import CONSEP
from resnet_unet import TwoEncodersOneDecoder
from utils.plotting import plot_loss

cuda_device = torch.device('cuda', 0)


def save_checkpoint(model, name, score=''):
    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, f'{name}{score:.3f}.pth')


def dice_loss_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Soft dice loss = 2*|A∩B| / |A|+|B|
    Note: x and target tensors should have values between 0 and 1
    """
    eps = 1e-7
    numerator = 2 * (x * target).sum((1, 2))
    denominator = (x + target).sum((1, 2))

    dice = 1 - (numerator + eps) / (denominator + eps)
    return dice


def train():
    model = TwoEncodersOneDecoder(resnet18, pretrained=True, out_channels=1)
    model.train()
    model.to(cuda_device)

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomPosterize(3),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.3, 0.3])
    ])

    dataset = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/train', mode='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)

    dataset_val = CONSEP('/itf-fi-ml/shared/courses/IN3310/mandatory2_data/val', mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=64, num_workers=10, pin_memory=True, drop_last=True)

    num_epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dataloader) * num_epochs))

    best_dice_score = 0
    best_epoch = 0
    bce_losses = []
    dice_losses = []
    for epoch in range(0, num_epochs):
        print(f'Epoch: {epoch}')
        epoch_start_time = time.time()
        for batch_idx, (x, h_x, y) in enumerate(dataloader, 1):
            # TODO: Step 1) Move x, h_x, and y to GPU
            x, h_x, y = x.to(cuda_device), h_x.to(cuda_device), y.to(cuda_device)
            # TODO: Step 2) Convert h_x to have 3 channels by repeating the 1 channel it has 3 times.
            #               Hint: You can use h_x.expand() function to do that without increasing memory usage
            #                     or use the .repeat() function
            h_x = h_x.expand(-1, 3, -1, -1)
            # TODO: Step 3) Run the model and get the outputs.
            
            # print(f"X shape: {x.shape}, H_X shape: {h_x.shape}, Y shape: {y.shape}")
            outputs = model(x, h_x).squeeze(1)
            # TODO: Step 4) a) Call the loss functions bce_loss_fn & dice_loss_fn. Add them to get the loss.
            #                  The loss should be a single number (not an array).
            #                   Hint: Use .mean() on dice_loss_fn's output
            #               b) Append the loss values to their respective lists for plotting.
            #                  Use .item() while appending the values.
            bce_loss = bce_loss_fn(outputs, y)
            dice_loss = dice_loss_fn(outputs.sigmoid(), y).mean()
            
            bce_losses.append(bce_loss.item())
            dice_losses.append(dice_loss.item())
            # TODO: Step 5) Run the backward() pass on the loss function
            total_loss = bce_loss + dice_loss
            total_loss.backward()
            # TODO: Step 6) Call the optimizer to update the model and then zero out the gradients.
            optimizer.step()
            optimizer.zero_grad()
            # The lines below prints loss values every 5 batches.
            # Uncomment them to see the loss go down during training.

            if batch_idx % 5 == 0 or batch_idx == len(dataloader) - 1:
                print(f' * {epoch}-{batch_idx:03}\t{round(bce_loss.item(), 6)} {round(dice_loss.item(), 6)} ', flush=True)

        scheduler.step()
        print(f'Epoch {epoch+1} took {timedelta(seconds=time.time() - epoch_start_time)}', flush=True)

        print('EVALUATING dice score on validation set')
        eval_start_time = time.time()
        dice_score_val = eval_dice_with_h_x(model, dataloader_val)
        print(f'Evaluation after epoch {epoch} took {timedelta(seconds=time.time() - eval_start_time)}', flush=True)
        if dice_score_val > best_dice_score:
            best_epoch = epoch
            best_dice_score = dice_score_val
            print('Saving model as a new best score has been achieved.')
            save_checkpoint(model, f'TwoEncodersOneDecoder_consep', dice_score_val.item())

    print(f'Best dice score achieved on validation dataset was {best_dice_score} for epoch {best_epoch}', flush=True)
    # Save loss values in case the plotting throws an error or you wanna plot with different parameters
    with open('bce_loss.npy', 'wb') as f:
        np.save(f, np.array(bce_losses))
    with open('dice_loss.npy', 'wb') as f:
        np.save(f, np.array(dice_losses))
    # Save loss plots
    print('Saving plots')
    plot_loss(bce_losses, 'bce_loss')
    plot_loss(dice_losses, 'dice_loss')
    print('Plots saved')


def eval_dice_with_h_x(model, dataloader):
    model.eval()
    dice = []
    for batch_idx, (x, h_x, y) in enumerate(dataloader):
        # TODO: Move (x, h_x, y) to cuda
        x, h_x, y = x.to(cuda_device), h_x.to(cuda_device), y.to(cuda_device)
        with torch.no_grad():
            # TODO: Step 1) Convert h_x to have 3 channels just like you did in the train() function
            h_x = h_x.expand(-1, 3, -1, -1)
            # TODO: Step 2) Run the model and store outputs in the variable out below
            out = model(x, h_x).squeeze(1)
            # TODO: Step 3) Convert the outputs to a binary mask as follows:
            #               a) Pass the output through the sigmoid function to get an output between 0 and 1
            #               b) Using 0.5 as the threshold, convert the values to 0 if they are < 0.5 and 1 if > 0.5
            
            out = torch.where(out.sigmoid() > 0.5, 1, 0)
            
        dice.append(dice_loss_fn(out, y))  # TODO: Replace None with the output of the dice_loss_fn called for the binary mask
    dice_score = 1 - torch.cat(dice, 0).mean()
    print(f'dice score (the higher the better): {dice_score:.4f}\n')
    model.train()
    return dice_score


if __name__ == '__main__':
    train()
