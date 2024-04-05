import argparse
from collections import defaultdict
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import os
import pickle
import torch
from transformers import ConvNextImageProcessor

transformation_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DataSplit():
    def __init__(self, data_path):
               
        self.train_images = []
        self.test_images = []
        self.val_images = []
        
        self.val_labels = []
        self.train_labels = []
        self.test_labels = []
       
        all_images, all_labels = self.read_data(data_path)
        
        # Split the data into lists of (image, label) tuples
        train, val, test = self.split_data(all_images, all_labels)
        
        self.ensure_disjointed(train[0], val[0], test[0])

        self.train_images, self.train_labels = train
        self.val_images, self.val_labels = val
        self.test_images, self.test_labels = test
        
    
    def read_data(self, data_path):
        all_data = defaultdict(list)
        images = []
        labels = []
        
        # Get all the files in the root directory
        for subdir, dirs, files in os.walk(data_path):
            if len(files) == 0:
                continue
            
            all_data[subdir] = [os.path.join(subdir, file) for file in files]
            category_name = subdir.split('/')[-1]
            
            images.extend([os.path.join(subdir, file) for file in files])
            labels.extend([category_name for _ in files])
        
        return images, labels
    
    def split_data(self, images, labels, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split the images and labels into train, validation and test sets
        """
        
        val_proportion = val_size / (val_size + test_size)
        
        train_img, other_img, train_labels, other_labels = train_test_split(images, 
                                                                            labels, 
                                                                            stratify=labels, 
                                                                            train_size=train_size,
                                                                            random_state=42,
                                                                            shuffle=True)
        
        val_img, test_img, val_labels, test_labels = train_test_split(other_img,
                                                                      other_labels,
                                                                      stratify=other_labels,
                                                                      train_size=val_proportion,
                                                                      random_state=42,)

        return (train_img, train_labels), (val_img, val_labels), (test_img, test_labels)

    def ensure_disjointed(self, train, val, test):
        # Throws an error on set intersection
        assert len(set(train) & set(val)) == 0, "Train and val overlap"
        assert len(set(train) & set(test)) == 0, "Train and test overlap"
        assert len(set(val) & set(test)) == 0, "Val and test overlap"
        
        return train, val, test
    
    
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images,
                 labels,
                 transform=None,
                 transform_probability=0.5,
                 device=None,
                 feature_extractor=None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.images = images
        self.labels = labels
        self.transform = transform
        self.transform_probability = transform_probability
        
        if feature_extractor:
            self.feature_extractor = feature_extractor
            self.rescale = None
        else:
            self.rescale = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor()
                                               ])
            self.feature_extractor = None   
    
        
        # Labels must be converted to number, but to ensure the same mapping we store the encoder
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)
        else:
            self.encoder = LabelEncoder()
            self.encoder.fit(labels)
            
            # Save the fitted encoder for future use
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)

            
    def int2label(self, int_label):
        if type(int_label) == list:
            return self.encoder.inverse_transform(int_label)
        
        return self.encoder.inverse_transform([int_label])[0]
    
    def label2int(self, label):
        if type(label) == list:
            return self.encoder.transform(label)
        
        return self.encoder.transform([label])[0]
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform and torch.rand(1) < self.transform_probability:
            image = self.transform(image)
            
        if self.rescale:
            image = self.rescale(image)
        
        # Encode to number
        label = self.label2int(label)
    
        # print(inputs['pixel_values'].shape)
        return image, label, image_name
    

    def collate_fn(self, batch):
        inputs, labels, names = zip(*batch)
        
        if self.feature_extractor:
            inputs = self.feature_extractor(images=inputs, return_tensors="pt")
        else:
            inputs = {'pixel_values': torch.stack(inputs)}
        
        # Move the tensors to the device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = np.array(labels)
        labels = torch.tensor(labels).to(self.device)
        
        return {'inputs': inputs, 'labels': labels, 'names': names}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()
    
    rootdir = args.data_path

    print(f"Reading data from {rootdir}...")
    data = DataSplit(rootdir)
    
    print(f" * train size: {len(data.train_images)}")
    print(f" * val size: {len(data.val_images)}")
    print(f" * test size: {len(data.test_images)}\n")
    
    print(f"Creating datasets and dataloaders with batch size {args.batch_size}...")
    
    train_data = ImageDataset(data.train_images,
                              data.train_labels,
                              transform=transformation_1)
    
    val_data = ImageDataset(data.val_images,
                            data.val_labels,
                            transform=transformation_1)
    
    test_data = ImageDataset(data.test_images,
                             data.test_labels,
                             transform=transformation_1)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=args.batch_size,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    
    print(f" * train loader size: {len(train_loader)}")
    print(f" * val loader size: {len(val_loader)}")
    print(f" * test loader size: {len(test_loader)}\n")
    
    for data in val_loader:
        continue
    
    print(f"Dataloader iterated through successfully!")
