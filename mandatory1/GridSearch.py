import torch
from torch.utils.data import DataLoader

from Data import DataSplit, ImageDataset, transformation_1


class GridSearch:
    def __init__(self,
                 model_class,
                 data_dir,
                 lr_values=[0.001,5e-5],
                 batch_size_values=[8, 16],
                 optimizers=[torch.optim.AdamW],
                 epochs_values=[10],
                 transforms=[None, transformation_1],
                 early_stopping=True,
                 patience=5,
                 save_results=True,
                 verbose=False,
                 device=None):
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.best_model = None
        self.best_map = 0
        
        self.model_class = model_class
        self.data_dir = data_dir
        self.lr_values = lr_values
        self.batch_size_values = batch_size_values
        self.epochs_values = epochs_values
        self.verbose = verbose
        self.save_results = save_results
        self.early_stopping = early_stopping
        self.patience = patience
        self.transforms = transforms
        self.optimizers = optimizers
        
        self.start_search()
        
    
    def start_search(self):
        print(f"\n\n{'='*100}\n    STARTING A GRID SEARCH...\n{'='*100}\n\n")
        for optimizer in self.optimizers:
            for lr in self.lr_values:
                for epoch in self.epochs_values:
                    for batch_size in self.batch_size_values:
                        for transform in self.transforms:
                            print(f"\n\n+{'-'*85}+")
                            print(f"  Testing for:\n  lr={lr}, batch={batch_size}, epochs={epoch},"+\
                                  f" transform: {transform is not None}, optimizer: {optimizer.__name__}...")
                            print(f"+{'-'*85}+\n")
                            
                            train, val, test = self.create_data(batch_size=batch_size,
                                                                transform=transform)
                            
                            model = self.model_class(lr=lr, device=self.device, optimizer=optimizer)
                            model.train(train,
                                        val,
                                        epochs=epoch,
                                        verbose=self.verbose,
                                        early_stopping=self.early_stopping,
                                        patience=self.patience,
                                        save_model=False,
                                        save_plots=True)
                            
                            metrics = model.evaluate(test)
                            map_score = metrics['map_score']
                            
                            if map_score > self.best_map:
                                self.best_map = map_score
                                self.best_model = model
                                print(f" NEW BEST MODEL FOUND WITH MAP SCORE: {map_score:.4}")
                        
        return self.best_model, self.best_map
    
    
    def create_data(self, batch_size, transform=None):
        data = DataSplit(self.data_dir)

        train_data = ImageDataset(data.train_images,
                                    data.train_labels,
                                    transform=transform,
                                    device=self.device)
        val_data = ImageDataset(data.val_images,
                                data.val_labels,
                                transform=transform,
                                device=self.device)
        test_data = ImageDataset(data.test_images,
                                    data.test_labels,
                                    transform=transform,
                                    device=self.device)

        train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=val_data.collate_fn)
        test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=test_data.collate_fn)
    
        return train_loader, val_loader, test_loader