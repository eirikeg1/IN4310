
import argparse
from Data import DataSplit, ImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()

    rootdir = args.data_path

    print(f"Creating dataset from {rootdir}...")
    data = DataSplit(rootdir)

    train_data = ImageDataset(data.train_images,
                                data.train_labels)
    val_data = ImageDataset(data.val_images,
                            data.val_labels)
    test_data = ImageDataset(data.test_images,
                                data.test_labels)

    train_loader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_data,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=True)

    print(f" * train loader size: {len(train_loader)}")
    print(f" * val loader size: {len(val_loader)}")
    print(f" * test loader size: {len(test_loader)}")
    
    
    
