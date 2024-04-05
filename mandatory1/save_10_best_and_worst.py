from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from Data import DataSplit, ImageDataset
from ResNetModel import ResNetModel


def save_10_best_and_worst(
    model,
    test_loader,
    int2label=None,
    save_dir="best_worst_images",
):
    # logits = list of logits (index represents class)
    # logits_per_image = {image_name : logits}
    logits_per_image = model.image_class_probabilities(test_loader)
    
    
    # logits_per_class = {class_index : {image_name : logit}}
    logits_per_class = defaultdict(dict)
    
    for image_name, logits in logits_per_image.items():
        for class_index, logit in enumerate(logits):
            logits_per_class[class_index][image_name] = logit
            
            
    
    # Convert int labels to class names
    logits_per_class = {
        int2label(int_label): scores
        for int_label, scores in logits_per_class.items()
    }
    
    
    # Sort logits
    logits_per_class = {
        class_name: sorted(scores.items(), key=lambda x: x[1])
        for class_name, scores in logits_per_class.items()
    }
    
   
    
    # Save 10 best and worst for each class
    for class_name, scores in logits_per_class.items():
        for i, (image_name, score) in enumerate(scores[:10]):
            image = Image.open(image_name)
            image.save(f"{save_dir}/{class_name}_worst_{i + 1}.png")
        
        for i, (image_name, score) in enumerate(scores[-10:]):
            image = Image.open(image_name)
            image.save(f"{save_dir}/{class_name}_best_{10-i}.png")
    


if __name__ == '__main__':
    
    model = ResNetModel(state_dict=torch.load("models/ResNet18_epoch_8_vloss_0.3404.pt"))
    
    data = DataSplit("data")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    
    test_data = ImageDataset(data.test_images,
                             data.test_labels,
                             feature_extractor=feature_extractor)
    
    test_loader = DataLoader(test_data,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=test_data.collate_fn)

    save_10_best_and_worst(
        model,
        test_loader,
        int2label=test_data.int2label,
        save_dir="best_worst_images"
    )
