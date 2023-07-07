from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob

class TrainDataset(Dataset):
    def __init__(self, root, tfms=None):
        super(TrainDataset, self).__init__()
        self.tfms = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
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
        imgs = glob.glob(os.path.join(root, 'train', 'good', '*'))
        return imgs

class TestDataset(Dataset):
    def __init__(self, root, tfms=None):
        super(TestDataset, self).__init__()
        self.tfms = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
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
            im_paths = sorted(glob.glob(os.path.join(root, 'test', cls, '*')))
            ims.extend(im_paths)
            if cls == 'good': labels.extend([0] * len(im_paths))
            else: labels.extend([1] * len(im_paths))
        return ims, labels