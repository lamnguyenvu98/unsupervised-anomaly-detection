from torch.utils.data import Dataset, Sampler
import torch
from torchvision import transforms
from PIL import Image
import random
import glob
import os

class TrainDataset(Dataset):
    def __init__(self, root, tfms=None, ignore_class=['transistor']):
        super(TrainDataset, self).__init__()
        self.tfms = transforms.Compose(
            [
                transforms.Resize([256, 256], Image.ANTIALIAS),
                transforms.ToTensor(),
            ]
        ) if tfms is None else tfms
        # if our test object (e.g: we want to test "transistor" is normal or not, 
        # "transistor" shouldn't be in the training set)
        # if "transistor" was in our root directory
        # we have to ignore it (e.g: self.ignore_class = ['pill'])
        # we can ignore more than one object, just throw them in self.ignore_class
        # if test object was not in root directory then, we have nothing to worry about.
        self.ignore_class = ignore_class
        self.total_labels = self.read_data(root)
    
    def __getitem__(self, idx):
        img = Image.open(self.total_labels[idx][0]).convert('RGB')
        img = self.tfms(img)
        return img
    
    def __len__(self):
        return len(self.total_labels)
    
    def get_labels(self):
        return [label[1] for label in self.total_labels]
    
    def read_data(self, root):
        total_labels = list()
        # root directory will contain a bunch of different objects class (e.g: bottle, transistor, screw,...)
        classes = os.listdir(root)
        # remove unnecessary files
        classes = sorted([cls for cls in classes if os.path.isdir(os.path.join(root, cls))])
        # remove test objects from our training object
        if self.ignore_class:
            for ignore_cls in self.ignore_class:
                if ignore_cls in classes:
                    classes.remove(ignore_cls)
        # get all paths of normal image of each object in train set
        for cls in classes:
            cls_imgs = glob(os.path.join(root, cls, 'train', 'good', '*.png'))
            labels = [tuple([img_path, cls]) for img_path in cls_imgs]
            total_labels.extend(labels)
        return total_labels
    
class TaskTrainSampler(Sampler):
    def __init__(self,  dataset,
                        n_shot: int,
                        batch_size: int):
        super().__init__(data_source=None)
        # size of support set
        self.n_shot = n_shot
        # batch_size
        self.batch_size = batch_size
        self.cls_to_idx = {}
        # mapping indices of image to its associated class object
        for index, cls in enumerate(dataset.get_labels()):
            if cls in self.cls_to_idx.keys():
                self.cls_to_idx[cls].append(index)
            else:
                self.cls_to_idx[cls] = [index]
        
    def __iter__(self):
        # iterate through each class object
        for cls, cls_img_list in self.cls_to_idx.items():
            # iterate through its image indices
            for i in range(0, len(cls_img_list) - self.batch_size + 1, self.batch_size):
                # build a query indices with size of self.batch_size
                query_indices = cls_img_list[i:i+self.batch_size]
                # build a support indices and it should not contain any query indices 
                list_support_indices = list(set(cls_img_list) - set(query_indices))
                # randomly pick some of indices from support list and concat them to query indices
                # then return to DataLoader
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
    def __init__(self, test_path, tfms=None):
        super(TestDataset, self).__init__()
        # test_path: path of test product
        # test_path should be a folder and inside of it
        # there should be a "train" folder and a "test" folder
        # but we will ignore the the "train" folder though
        self.test_path = test_path
        # resize image and convert it to tensor
        self.tfms = transforms.Compose(
            [
                transforms.Resize([224, 224], Image.ANTIALIAS),
                transforms.ToTensor(),
            ]
        ) if tfms is None else tfms
        
        self.total_labels = self.read_data(test_path)
    
    def __getitem__(self, idx):
        # read each image from its path and convert to RGB image
        img = Image.open(self.total_labels[idx][0]).convert('RGB')
        # transform this image (Resize it and convert it to torch.Tensor)
        img = self.tfms(img)
        # extract label (0 or 1) associated to this image
        label = self.total_labels[idx][-1]
        # return image tensor and its label
        return img, label
    
    def __len__(self):
        return len(self.total_labels)
        
    def read_data(self, test_path):
        total_labels = list()
        cls_idx = None
        # Iterate through child folders in test folder
        # test folder can contain "anomaly" folder and "good" folder
        # "anomaly" folder contains defect images, while "good" folder contain normal images
        for anomaly_label in os.listdir(os.path.join(test_path, 'test')):
            # anomaly_label can be "anomaly" or "good" (we have 2 labels "good" and "anomaly")
            cls_imgs = sorted(glob(os.path.join(test_path, 'test', anomaly_label, '*')))
            # if we are currently at "good" folder then cls_idx = 0
            # if we are currently at other folders (e.g: "anomaly" folder) then cls_idx = 1
            if anomaly_label == 'good': cls_idx = 0
            else: cls_idx = 1

            labels = [tuple([img_path, cls_idx]) for img_path in cls_imgs]
            total_labels.extend(labels)
        
        # return a list of tuples, which contain each path of the image and its associated label
        return total_labels
