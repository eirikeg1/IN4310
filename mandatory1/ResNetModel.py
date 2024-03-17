from collections import defaultdict
import time
from matplotlib import pyplot as plt
from sklearn import feature_extraction
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoFeatureExtractor
import torch
import torch.nn as nn
from datasets import load_dataset


class ResNetModel:
    def __init__(self, model_version="microsoft/resnet-18",
                 lr=5e-5,
                 optimizer=torch.optim.AdamW,
                 device=None):
        
        self.train_loss_values = defaultdict(list)
        self.val_loss_values = defaultdict(list)
        self.metrics = {}
        
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
    
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                     step_size=7,
        #                                                     gamma=0.1)

            
        
 
    ## TRAINING METHODS
    
    def train_epoch(self, train_loader):
        self.model.train()
        loss_values = []
        
        num_batches = len(train_loader)
        
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
            # self.lr_scheduler.step()
            
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
        
    
    def train(self,
              train_loader,
              val_loader = None,
              epochs=10,
              verbose=True,
              early_stopping=True,
              patience=3,
              save_model=True,
              save_plots=True,
              save_checkpoints=False,
              model_folder="models",):
        
        self.model_name = f"ResNet18"
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            self.train_loss_values[self.model_name].append(train_loss)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            if not val_loader:
                continue
                
            val_loss = self.val_loss(val_loader)
            self.val_loss_values[self.model_name].append(val_loss)
            
            map_score, acc, f1, prec, rec = self.calculate_metrics(val_loader)
            
            self.metrics_string = f"_epoch_{epoch}_vloss_{val_loss:.4}_map_{map_score:.4}"
            
            if verbose:
                evaluation_string = (f"Epoch {epoch+1}\n{'-'*25}\n"
                                     f" * train-loss: {train_loss:.4}\n"
                                     f" * val-loss: {val_loss:.4}\n"
                                     f" * accuracy: {acc:.4}\n"
                                     f" * f1: {f1:.4}\n"
                                     f" * precision: {prec:.4}\n"
                                     f" * recall: {rec:.4}\n\n")
                print(evaluation_string)             
            
            # Early stopping
            if save_model and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_checkpoints:
                    self.save_checkpoint(epoch, val_loss, map_score)
            else:
                patience_counter += 1
            
            if early_stopping and patience_counter >= patience:
                print(f'Early Stopping triggered after epoch {epoch+1}')
                break
        
        
        print(f"Finished training after {time.time() - start_time:.2f} seconds\n")
        
        if save_model:  
            torch.save(self.model.state_dict(),
                       f"{model_folder}/{self.model_name}{self.metrics_string}.pt")
            
        if save_plots:
            self.save_plots()
                       
            
                
    ## EVALUATION METHODS
    
    def calculate_metrics(self, data_loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        metrics = {}
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                
                inputs = batch['inputs']
                labels = batch['labels']
                
                outputs = self.model(**inputs)
                
                preds = torch.argmax(outputs.logits, dim=1)
                
                # FIX CODE BELOW
                all_labels.extend(labels.tolist())
                all_predictions.extend(preds.tolist())
        
        # Binarize the labels and predictions
        classes = list(set(all_labels))
        all_labels_bin = label_binarize(all_labels, classes=classes)
        all_predictions_bin = label_binarize(all_predictions, classes=classes)
        
        # Compute the average precision for each class
        ap_per_class = [average_precision_score(all_labels_bin[:, i], all_predictions_bin[:, i]) 
                        for i in range(len(classes))]
        
        report = classification_report(all_labels,
                                       all_predictions,
                                       output_dict=True,
                                       zero_division=0)

        # Compute the mean average precision
        metrics["map_score"] = sum(ap_per_class) / len(ap_per_class)
        metrics["accuracy"] = report['accuracy']
        metrics["f1_score"] = report['macro avg']['f1-score']
        metrics["precision"] = report['macro avg']['precision']
        metrics["recall"] = report['macro avg']['recall']
        
        self.metrics = metrics
        
        return metrics
                


     # SAVE RESULT METHODS      
                
    def save_checkpoint(self,
                        epoch,
                        val_loss,
                        map_score,
                        model_folder="models",
                        model_name="ResNet18"):
        
        torch.save(self.model.state_dict(),
                   f"{model_folder}/checkpoint_epoch_{epoch}_vloss_{val_loss:.4}_map_{map_score:.4}.pt")
        
    def save_plots(self):
        for model_name in self.val_loss_values.keys():
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot validation loss on first subplot
            axs[0].plot(self.val_loss_values[model_name], label=f"{model_name} Validation Loss")
            axs[0].legend()
            axs[0].set_title("Validation Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            
            # Plot training loss on the second subplot
            axs[1].plot(self.train_loss_values[model_name], label=f"{model_name} Training Loss")
            axs[1].legend()
            axs[1].set_title("Training Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss")
            
            plt.tight_layout() # Adjust layout so title/labels don't overlap
            plt.savefig(f"plots/{model_name}_losses.png")
            plt.close(fig)


