from sklearn import feature_extraction
from sklearn.metrics import classification_report
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoFeatureExtractor
import torch
import torch.nn as nn
from datasets import load_dataset


class ResNetModel:
    def __init__(self, model_version="microsoft/resnet-18", lr=0.001):
        
        self.self.model = ResNetForImageClassification.from_pretrained(model_version)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, 
                                                   size_average=None,
                                                   ignore_index=-100,
                                                   reduce=None,
                                                   reduction='mean')
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
 
 
    ## TRAINING METHODS
    
    def train_epoch(self, train_loader):
        self.model.train()
        for i, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            loss = self.model(**batch).loss

            # backward pass
            loss.backward()

            # update weights
            self.optimizer.step()
            self.lr_scheduler.step()
            
            
            return loss.item()
        
    
    def train(self, train_loader, val_loader, epochs=10, verbose=True):
        
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader, val_loader)
            
            
            if verbose:
                acc, f1, prec, rec = self.evaluate(val_loader)
                
                evaluation_string = f"""Epoch {epoch+1}, loss: {loss}\n{'-'*20}
                                        {'-'*20}
                                          * accuracy: {acc:.4}\n
                                          * f1: {f1:.4}\n
                                          * precision: {prec:.4}\n
                                          * recall: {rec:.4}\n\n
                                     """
                print(evaluation_string)
                
                
                
    ## EVALUATION METHODS
    
    def evaluate(self, val_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        report = classification_report(all_labels, all_predictions, output_dict=True)
        accuracy = report['accuracy']
        f1_score = report['macro avg']['f1-score']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        
        return accuracy, f1_score, precision, recall
                



