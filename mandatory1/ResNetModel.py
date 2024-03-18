from collections import defaultdict
import copy
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sklearn
from sklearn import feature_extraction
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoFeatureExtractor
import torch
import torch.nn as nn
from datasets import load_dataset


class ResNetModel:
    def __init__(self,
                 model="microsoft/resnet-18",
                 lr=5e-5,
                 optimizer=torch.optim.AdamW,
                 num_labels=6,
                 state_dict=None,
                 device=None):
        
        self.train_loss_values = defaultdict(list)
        self.val_loss_values = defaultdict(list)
        self.metrics = {}
        self.metrics_string = ""
        self.model_info = "ResNet_not_fine_tuned"
        self.softmax_file = None
        self.activations = None
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(model, str):
            self.model = ResNetForImageClassification.from_pretrained(model)
            # update number of out features
            # print(self.model)
            self.model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.model.classifier[1].in_features, num_labels)
            )
            # self.model.classifier = nn.Linear(self.model.classifier[1].in_features, num_labels)
        else:
            self.model = model
        
        if state_dict:
            self.model.load_state_dict(state_dict)
            self.model_info = "ResNet_fine_tuned"
            
            
        self.model.to(self.device)
        self.best_state = self.model.state_dict()

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
              model_folder="models",
              detailed_metrics=False):
        
        self.model_name = f"ResNet18"
        start_time = time.time()
        t_loss = []
        v_loss = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            if not val_loader:
                continue
                
            val_loss = self.val_loss(val_loader)
            
            t_loss.append(train_loss)
            v_loss.append(val_loss)
            
            self.metrics_string = f"epoch_{epoch+1}_vloss_{val_loss:.4}"
            
            evaluation_string = (f"Epoch {epoch+1}\n{'-'*25}\n"
                                 f" * train-loss: {train_loss:.4}\n"
                                 f" * val-loss: {val_loss:.4}")
            
            if detailed_metrics:
                metrics = self.calculate_metrics(val_loader)
                
                f1 = metrics['f1_score']
                acc = metrics['accuracy']
                prec = metrics['precision']
                rec = metrics['recall']
                map_score = metrics['map_score']
                
                evaluation_string += (f" * map: {map_score:.4}, "
                                      f"f1: {f1:.4}, "
                                      f"accuracy: {acc:.4}, "
                                      f"precision: {prec:.4}, "
                                      f"recall: {rec:.4}")
                
                self.metrics_string += f"_map_{map_score:.4}"
            
            if verbose:
                print(f"{evaluation_string}\n\n")             
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                
                patience_counter = 0
                
                if save_checkpoints:
                    self.save_checkpoint(epoch+1, val_loss, map_score)
            else:
                patience_counter += 1
            
            if early_stopping and patience_counter >= patience:
                print(f'Early Stopping triggered after epoch {epoch+1}')
                break
        
        
        print(f"\n{'-'*25}\n\nFinished training after {time.time() - start_time:.2f} seconds\n")
        
        self.model_info = f"{self.model_name}_{self.metrics_string}"
            
        self.train_loss_values[self.model_info] = t_loss
        self.val_loss_values[self.model_info] = v_loss
        
        # Updates the models state_dict to the last best state
        self.model.load_state_dict(self.best_state)
        
        if save_model:
            self.save_best()
            
        if save_plots:
            self.save_plots()
                       
            
                
    ## EVALUATION METHODS
    
    def calculate_metrics(self, data_loader, save_softmax=False):
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
                
                all_labels.extend(labels.tolist())
                all_predictions.extend(preds.tolist())
        
        # Binarize labels and predictions
        classes = list(set(all_labels))
        all_labels_bin = label_binarize(all_labels, classes=classes)
        all_predictions_bin = label_binarize(all_predictions, classes=classes)
        
        # Compute average precision for each class
        ap_per_class = [average_precision_score(all_labels_bin[:, i], all_predictions_bin[:, i]) 
                        for i in range(len(classes))]
        
        if save_softmax:
            self.softmax_scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            self.softmax_file = f"softmaxes/{self.model_info}.txt"
            metrics["softmax_scores"] = self.softmax_scores
            self.save_softmaxes()
            
        
        report = classification_report(all_labels,
                                       all_predictions,
                                       output_dict=True,
                                       zero_division=0)

        metrics["map_score"] = sum(ap_per_class) / len(ap_per_class)
        metrics["accuracy"] = report['accuracy']
        metrics["f1_score"] = report['macro avg']['f1-score']
        metrics["precision"] = report['macro avg']['precision']
        metrics["recall"] = report['macro avg']['recall']
        
        self.metrics = metrics
        
        
        
        return metrics
    
    def compare_softmaxes(self, file_path, tolerance=5e-5):
        saved_softmaxes = np.loadtxt(file_path, delimiter=",")
        current_softmaxes = self.softmax_scores
        
        print(f"\nComparing softmaxes from '{file_path}' with current softmaxes, with a tolerance of {tolerance}\n")
        
        assert np.allclose(saved_softmaxes,
                           current_softmaxes,
                           atol=tolerance), "Softmaxes are not equal"   
           
        print(f"Comparison successfull!\n")
        
        
    def getActivation(self, name):
        def hook(module, input, output):
            self.activations[name].append(output.cpu().detach().numpy())
        return hook
    
    def compute_feature_map_statistics(self, data_loader, selected_modules):
        
        self.activations = defaultdict(list)
        self.activation_labels = []
        
        # Attach hooks to selected modules
        for name, mod in selected_modules:
            mod.register_forward_hook(self.getActivation(name))
        
        num_of_images = 0
        
        # Iterate data loader
        for i, batch in enumerate(data_loader):
            inputs = batch['inputs']
            labels = batch['labels']
            
            self.model.eval()
            with torch.no_grad():
                _ = self.model(**inputs)
            
            self.activation_labels.extend(labels.cpu().tolist())
            
            # Only process the first 200 images (plus rest based on batch size)
            num_of_images += inputs['pixel_values'].shape[0]
            if num_of_images > 200:
                break
    
        # Calculate percentage of all positive activations
        all_activations_flattened = [np.ravel(activation) 
                                     for activation in self.activations.values()]
        
        positive_values = [activation > 0 for activation in all_activations_flattened]

        percentages = [(np.sum(positive_values) / len(all_values)) * 100
                       for all_values, positive_values in zip(all_activations_flattened,
                                                              positive_values)]
        
        for name, percentage in zip(self.activations.keys(), percentages):
            print(f"Percentage of positive activations for {name}: {percentage:.2f}%")
            
        
        
    def PCA(self, data_loader, int2label, n_components=2):
        
        # Ensure model is in eval mode if this affects activation collection
        self.model.eval()
        
        activation_list = list(self.activations.values())
        
        def flatten_activations(activation):
            flat = []
            for act in activation:
                flat.extend(list(np.ravel(act)))
            return np.array(flat)
        
        flattened_activations = [flatten_activations(batch) for batch in activation_list]
        
        
        all_activations = np.vstack(flattened_activations)


        pca_decomp = sklearn.decomposition.PCA(n_components=n_components)
        pca_decomp.fit(all_activations)
        transformed_features = pca_decomp.transform(all_activations)
        
        labels = int2label(list(self.activation_labels))
        
        labels = np.array(labels)
        
        unique_labels = np.unique(labels)  # Get unique labels for legend

        # Prepare the plot
        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(labels)

        for unique_label in unique_labels:
            indices = labels == unique_label
            
            if np.sum(indices) == 0:
                continue 
            
            plt.scatter(transformed_features[indices, 0], 
                        transformed_features[indices, 1], 
                        label=unique_label)

        plt.title('2D PCA of Model Activations')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.savefig(f"pca/{self.model_info}.png")
        

     # SAVE RESULT METHODS    
     
    def save_best(self, model_folder="models"):
        print(f"Saving best model to '{model_folder}/{self.model_info}.pt'")
        
        torch.save(self.model.state_dict(),
                   f"{model_folder}/{self.model_info}.pt")  
                
    def save_checkpoint(self,
                        epoch,
                        val_loss,
                        map_score,
                        model_folder="models",
                        model_name="ResNet18"):
        print(f"Saving checkpoint after epoch {epoch+1} with val-loss: {val_loss:.4} and map: {map_score:.4}")
        torch.save(self.model.state_dict(), f"{model_folder}/checkpoint_{self.model_info}.pt")
       
    def save_softmaxes(self, save_folder="softmaxes"):
        lines = []
        for i in range(self.softmax_scores.shape[0]):
            lines.append(",".join([str(x) for x in self.softmax_scores[i]]))
        
        np.savetxt(f"{save_folder}/{self.model_info}.txt", lines, delimiter=",", fmt="%s")
        
    def save_plots(self):
        for model_name in self.val_loss_values.keys():
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot training loss on the second subplot
            x_values = range(1, len(self.train_loss_values[model_name]) + 1)
            axs[0].plot(x_values,
                        self.train_loss_values[model_name], 
                        label=f"{model_name} Training Loss")
            axs[0].legend()
            axs[0].set_title("Training Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Plot validation loss on first subplot
            x_values = range(1, len(self.val_loss_values[model_name]) + 1)
            axs[1].plot(x_values,
                        self.val_loss_values[model_name],
                        label=f"{model_name} Validation Loss")
            axs[1].legend()
            axs[1].set_title("Validation Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss")
            axs[1].xaxis.set_major_locator(MaxNLocator(integer=True)) # Only integer ticks
            
            plt.tight_layout() # Adjust layout so title/labels don't overlap
            plt.savefig(f"plots/{model_name}_losses.png")
            print(f"Plots saved")
            plt.close(fig)


