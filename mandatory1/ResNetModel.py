from sklearn import feature_extraction
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoFeatureExtractor
import torch
import torch.nn as nn
from datasets import load_dataset


class ResNetModel:
    def __init__(self, model_version="microsoft/resnet-18", lr=0.001, device=None):
        
        self.train_loss_values = []
        self.val_loss_values = []
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ResNetForImageClassification.from_pretrained(model_version).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, 
                                                   size_average=None,
                                                   ignore_index=-100,
                                                   reduce=None,
                                                   reduction='mean').to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=7,
                                                            gamma=0.1)

            
        
 
    ## TRAINING METHODS
    
    def train_epoch(self, train_loader):
        self.model.train()
        loss_values = []
        
        for i, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            inputs = batch['inputs']
            labels = batch['labels']
            
            outputs = self.model(**inputs)
            
            
            loss = self.criterion(outputs.logits, labels)
            
            # backward pass
            loss.backward()

            # update weights
            self.optimizer.step()
            self.lr_scheduler.step()
            
            loss_values.append(loss.item())
            
            
        return sum(loss_values) / len(loss_values)
    
    def val_loss(self, val_loader):
        self.model.eval()
        loss_values = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['inputs']
                labels = batch['labels']
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
        
            loss_values.append(loss.item())
        
        return sum(loss_values) / len(loss_values)
        
    
    def train(self, train_loader, val_loader = None, epochs=10, verbose=True):
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_loss_values.append(train_loss)
            
            if not val_loader:
                continue
                
            val_loss = self.val_loss(val_loader)
            self.val_loss_values.append(val_loss)
            
            if verbose:
                acc, f1, prec, rec = self.evaluate(val_loader)
                
                evaluation_string = (f"Epoch {epoch+1},\n"
                                     f" * train-loss: {train_loss}\n"
                                     f" * val-loss: {val_loss}\n"
                                     f" * accuracy: {acc:.4}\n"
                                     f" * f1: {f1:.4}\n"
                                     f" * precision: {prec:.4}\n"
                                     f" * recall: {rec:.4}\n\n")
                print(evaluation_string)
                
                
                
    ## EVALUATION METHODS
    
    def evaluate(self, val_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                
                inputs = batch['inputs']
                labels = batch['labels']
                
                outputs = self.model(**inputs)
                
                preds = torch.argmax(outputs.logits, dim=1)
                
                # FIX CODE BELOW
                all_labels.extend(labels.tolist())
                all_predictions.extend(preds.tolist())
                
                
        
        report = classification_report(all_labels, all_predictions, output_dict=True)
        accuracy = report['accuracy']
        f1_score = report['macro avg']['f1-score']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        
        return accuracy, f1_score, precision, recall
                



