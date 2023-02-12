from torch.utils.data import Dataset, Sampler
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import os

class TrainDataset(Dataset):
    def __init__(self, root, tfms=None):
        super(TrainDataset, self).__init__()
        self.tfms = transforms.Compose(
            [
                transforms.Resize([224, 224], Image.ANTIALIAS),
                transforms.ToTensor(),
            ]
        ) if tfms is None else tfms

        self.datas = self.read_data(root)
    
    def __getitem__(self, idx):
        img = Image.open(self.datas[idx]).convert('RGB')
        img = self.tfms(img)
        return img
    
    def __len__(self):
        return len(self.datas)
    
    def get_labels(self):
        return np.arange(len(self.datas)).tolist()
    
    def read_data(self, root):
        imgs = glob(os.path.join(root, 'train', 'good', '*'))
        return imgs
    
class TaskTrainSampler(Sampler):
    def __init__(self,  dataset,
                        n_shot: int,
                        batch_size: int):

        super().__init__(data_source=None)
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.data_indices = dataset.get_labels()
        
    def __iter__(self):
        for i in range(0, len(self.data_indices) - self.batch_size + 1, self.batch_size):
            query_indices = self.data_indices[i:i+self.batch_size]
            list_support_indices = list(set(self.data_indices) - set(query_indices))
            yield query_indices + random.sample(list_support_indices, self.n_shot * self.batch_size)


    def episodic_collate_fn(self, input_data):
        all_images = torch.cat([x.unsqueeze(0) for x in input_data])

        all_images = all_images.reshape(
            (self.batch_size * (1 + self.n_shot), *all_images.shape[1:])
        )

        query_images = all_images[:self.batch_size, ...]

        support_images = all_images[self.batch_size:, ...].view(self.batch_size, self.n_shot, *all_images.shape[1:])

        return query_images, support_images

class TestDataset(Dataset):
    def __init__(self, root, tfms=None):
        super(TestDataset, self).__init__()
        self.tfms = transforms.Compose(
            [
                transforms.Resize([224, 224], Image.ANTIALIAS),
                transforms.ToTensor(),
            ]
        ) if tfms is None else tfms

        self.datas, self.labels = self.read_data(root)
    
    def __getitem__(self, idx):
        img = Image.open(self.datas[idx]).convert('RGB')
        img = self.tfms(img)
        label = self.labels[idx]
        return img, label
    
    def __len__(self):
        return len(self.datas)
    
    def read_data(self, root):
        ims, labels = [], []
        for cls in os.listdir(os.path.join(root, 'test')):
            im_paths = sorted(glob(os.path.join(root, 'test', cls, '*')))
            ims.extend(im_paths)
            if cls == 'good': labels.extend([0] * len(im_paths))
            else: labels.extend([1] * len(im_paths))
        return ims, labels