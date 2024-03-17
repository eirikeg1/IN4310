
import argparse
from Data import DataSplit, ImageDataset, transformation_1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from GridSearch import GridSearch
from ResNetModel import ResNetModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # print(f"\nCreating dataset from './{root_dir}'...\n{'='*40}\n")
    # data = DataSplit(args.root_dir)

    # train_data = ImageDataset(data.train_images,
    #                             data.train_labels)
    # val_data = ImageDataset(data.val_images,
    #                         data.val_labels)
    # test_data = ImageDataset(data.test_images,
    #                             data.test_labels)

    # train_loader = DataLoader(train_data,
    #                         batch_size=args.batch_size,
    #                         shuffle=True,
    #                         collate_fn=train_data.collate_fn)
    # val_loader = DataLoader(val_data,
    #                         batch_size=args.batch_size,
    #                         shuffle=True,
    #                         collate_fn=val_data.collate_fn)
    # test_loader = DataLoader(test_data,
    #                         batch_size=args.batch_size,
    # #                         shuffle=True,
    #                         collate_fn=test_data.collate_fn)

    # print(f" * train loader size: {len(train_loader)}")
    # print(f" * val loader size: {len(val_loader)}")
    # print(f" * test loader size: {len(test_loader)}")
    
    # model = ResNetModel()
    # print(f"\n\nTraining model...\n{'='*40}\n")
    # model.train(train_loader, val_loader, epochs=10, verbose=args.verbose)
    
    
    # The three hyperparameter settings I tested with was different learning rates, batch sizes and if to transform
    
    transformation_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    lr_rates = [0.001, 5e-5]
    optimizers = [torch.optim.AdamW, torch.optim.SGD]
    transformations = [None, transformation_1]
    
    grid_search = GridSearch(ResNetModel,
                             args.root_dir,
                             lr_values=lr_rates,
                             optimizers=optimizers,
                             transforms=transformations,
                             batch_size_values=[8],
                             epochs_values=[3],
                             early_stopping=True,
                             patience=5,
                             save_results=True,
                             verbose=False)
    
    
    
    
    