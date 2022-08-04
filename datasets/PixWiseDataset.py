import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms


class PixWiseDataset(Dataset):
    """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way

    Args:
        root_dir (string): Root directory path
        csv_file (string): csv file to dataset annotation
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """

    def __init__(self, mode, root_dir, csv_file, map_size, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.map_size = map_size
        self.transform = transform
        self.mode = mode
        if self.mode=='CelebA_Spoof_train' or self.mode=='CelebA_Spoof_test':
            self.root_dir = '/data/f77bb6f4e7214dccaefc28665eb2da08/CelebA_Spoof_crop/'
        root = os.path.join(self.root_dir, csv_file)
        print('Dataset path ls: ', root)
        with open(root, 'r') as f:
            if self.mode=='train' or self.mode=='CelebA_Spoof_train':
                self.lines = f.readlines()
                # self.lines = f.readlines()[:10000000]
#                 self.lines = f.readlines()[:1000000]
                # self.lines = f.readlines()[:2048]
#                 self.lines = f.readlines()[:512]
#                 self.lines = f.readlines()[:16]
            elif self.mode=='test' or self.mode=='CelebA_Spoof_test':
                self.lines = f.readlines()
#                 self.lines = f.readlines()[6500000:7000000]
                # self.lines = f.readlines()[:200000]
#                 self.lines = f.readlines()[:10000]
                # self.lines = f.readlines()[:2048]
#                 self.lines = f.readlines()[:512]
#                 self.lines = f.readlines()[:16]
                
        if smoothing:
            self.label_weight = 0.99
        else:
            self.label_weight = 1.0

    def __getitem__(self, index):
        """ Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
            img (PIL Image): 
            mask: output map (14x14)
            label: 1 (genuine), 0 (fake) 
        """
        line = self.lines[index]
        if len(line.strip('\n').split()) == 4:
            image_path, eyeopenrate, eyelabel, label = line.strip('\n').split()
            if not os.path.exists(image_path):    
                image_path = self.lines[0].strip('\n').split(' ')[0]
                
        if len(line.strip('\n').split()) == 2:
            image_path, label = line.strip('\n').split()
            image_path = os.path.join(self.root_dir, image_path)
            if not os.path.exists(image_path):    
                image_path = self.lines[0].strip('\n').split(' ')[0]
                image_path = os.path.join(self.root_dir, image_path)
                
#         if index%2==0:
#             image_path = '/home/semtp/notebooks/live.jpg'   
#             label == 1
#         if index%2==1:
#             image_path = '/home/semtp/notebooks/spoof.jpg' 
#             label == 0
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = np.expand_dims(float(label), axis=0)
        if self.mode=='test' or self.mode=='CelebA_Spoof_test':
            if label == 1:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * self.label_weight
            else:
                mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1.0 - self.label_weight)            
            return image, mask, label, image_path # tensors, numpy arrays, numbers, dicts or lists; 
        
        return image, label, image_path
    def __len__(self):
#         print('Dataset Len is: ', len(self.lines))
        return len(self.lines)
    
    
# class PixWiseDataset(Dataset):
#     """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way
#     Args:
#         root_dir (string): Root directory path
#         csv_file (string): csv file to dataset annotation
#         map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
#         transform: A function/transform that takes in a sample and returns a transformed version
#         smoothing (bool): Use label smoothing
#     """
#     def __init__(self, root_dir, csv_file, map_size, transform=None, smoothing=True):
#         super().__init__()
#         self.root_dir = root_dir
#         all_data  = pd.read_csv(os.path.join(root_dir, csv_file))
#         data_used = all_data.iloc[:int(len(all_data))]
#         # self.data = data_used
#         self.data=data_used
#         self.map_size = map_size
#         self.transform = transform
        
#         if smoothing:
#             self.label_weight = 0.99
#         else:
#             self.label_weight = 1.0
#     def __getitem__(self, index):
#         """ Get image, output map and label for a given index
#         Args:
#             index (int): index of image
#         Returns:
#             img (PIL Image): 
#             mask: output map (14x14)
#             label: 1 (genuine), 0 (fake) 
#         """
#         img_name = self.data.iloc[index, 0]
#         img_name = os.path.join(self.root_dir, img_name)
#         img = Image.open(img_name)
#         label = self.data.iloc[index, 1].astype(np.float32)
#         label = np.expand_dims(label, axis=0)
#         if label == 1:
#             mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * self.label_weight
#         else:
#             mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1.0 - self.label_weight)
#         if self.transform:
#             img = self.transform(img)
#         print('label-3:', label.shape)
#         print('label-3:', label)
#         return img, mask, label,img_name # label.shape (1,) , value:[0.]
#     def __len__(self):
#         return len(self.data)
    