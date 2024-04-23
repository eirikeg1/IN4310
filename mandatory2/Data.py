from collections import defaultdict
from sklearn.model_selection import train_test_split


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


