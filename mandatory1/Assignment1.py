
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
           root_dir='data'):
    # Task 1
    transformation_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ])
    
    lr_rates = [1e-4]
    optimizers = [torch.optim.AdamW]
    transformations = [transformation_1]
    
    # Store best model
    if not skip_grid_search:
        grid_search = GridSearch(ResNetModel,
                                root_dir,
                                lr_values=lr_rates,
                                optimizers=optimizers,
                                transforms=transformations,
                                transform_probability=0.5,
                                batch_size_values=[16],
                                epochs_values=[8],
                                early_stopping=True,
                                patience=3,
                                save_results=True,
                                verbose=verbose,
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
    
    
def task_3(model, val_data, test_loader):
    
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=val_data.collate_fn)
    
    base_model = ResNetModel()
    
    print(f"\n\nPerforming PCA on the feature maps\n\n")
    model.PCA(val_loader, val_data.int2label, n_components=2)
    
    task_2(base_model, test_loader)  
    base_model.PCA(val_loader, val_data.int2label, n_components=2)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--skip-grid-search', action='store_true')
    args = parser.parse_args()
    
    best_model = "models/ResNet18_epoch_8_vloss_0.3404.pt"
    # model = ResNetModel()
    # print(f"\n\nTraining model...\n{'='*40}\n")
    # model.train(train_loader, val_loader, epochs=10, verbose=args.verbose)
    
    # The three hyperparameter settings I tested with was different learning rates, batch sizes and
    # if to rotate/flip the image
    
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    
    model = ResNetModel(model="microsoft/resnet-18")
    
    # Load saved model if not doing task 1 or if skipping grid search
    if args.skip_grid_search or args.task > 1:
        state_dict = torch.load(best_model)
        model.model.load_state_dict(state_dict)
    
    data = DataSplit(args.root_dir)
    
    train_data = ImageDataset(data.train_images,
                              data.train_labels,
                              feature_extractor=feature_extractor)
    
    val_data = ImageDataset(data.val_images,
                            data.val_labels,
                            feature_extractor=feature_extractor)
    
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
               verbose=args.print,
               root_dir=args.root_dir)
        
    if args.task > 1 or args.task == 0:
        task_2(model, test_loader)
    
    if args.task == 3 or args.task == 0:
        task_3(model, val_data, test_loader)
    
    
    
    
    