
import argparse

import numpy as np
from Data import DataSplit, ImageDataset, transformation_1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from GridSearch import GridSearch
from ResNetModel import ResNetModel

def task_1(model,
           test_loader,
           skip_grid_search=False,
           verbose=False,
           root_dir='data',
           batch_size=16):
    # Task 1
    transformation_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ])
    
    lr_rates = [1e-4]
    optimizers = [torch.optim.AdamW, torch.optim.SGD]
    transformations = [None, transformation_1]
    
    # Store best model
    if not skip_grid_search:
        grid_search = GridSearch(ResNetModel,
                                root_dir,
                                lr_values=lr_rates,
                                optimizers=optimizers,
                                transforms=transformations,
                                transform_probability=0.5,
                                batch_size_values=[16],
                                epochs_values=[10],
                                early_stopping=True,
                                patience=3,
                                save_results=True,
                                verbose=True,
                                detailed_metrics=False)
    
    # Load best model
    print(f"Saving softmax values, predict once more and then compare the softmax values...")
    model.calculate_metrics(test_loader, save_softmax=True)
    softmax_file = model.softmax_file
    
    model.calculate_metrics(test_loader, save_softmax=False)
    model.compare_softmaxes(softmax_file)
    
    
def task_2(model, test_loader):    
    
    selected_indexes = [2, 30, 55, 80, 110]
    
    selected_modules = [mod for i, mod in enumerate(model.model.named_modules())
                        if i in selected_indexes]
    
    model.compute_feature_map_statistics(test_loader, selected_modules)    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--skip-grid-search', action='store_true')
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    
    # model = ResNetModel()
    # print(f"\n\nTraining model...\n{'='*40}\n")
    # model.train(train_loader, val_loader, epochs=10, verbose=args.verbose)
    
    # The three hyperparameter settings I tested with was different learning rates, batch sizes and
    # if to rotate/flip the image
    
    pretrained_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    
    model = ResNetModel(model=pretrained_model)
    
    data = DataSplit(args.root_dir)

    test_data = ImageDataset(data.test_images,
                             data.test_labels,
                             feature_extractor=feature_extractor)

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=test_data.collate_fn)
    
    if args.task == 1 or args.task == 0:
        task_1(model,
               test_loader,
               skip_grid_search=args.skip_grid_search,
               verbose=args.verbose,
               root_dir=args.root_dir,
               batch_size=args.batch_size)
        
    if args.task == 2 or args.task == 0:
        task_2(model, test_loader)
    
    
    
    # Task 2
    
    
    
    