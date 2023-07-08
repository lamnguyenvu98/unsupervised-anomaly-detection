from torchvision import transforms
from PIL import Image
import random
from glob import glob
import os
import torch

class GenerateSupportSet():
    def __init__(self, test_path, num_test, n_shot, tfms=None):
        self.test_path = test_path
        self.n_test = num_test
        self.n_shot = n_shot
        self.tfms = transforms.Compose(
            [
                transforms.Resize([224, 224], Image.ANTIALIAS),
                transforms.ToTensor(),
            ]
        ) if tfms is None else tfms
        self.img_paths = self.read_data(test_path)

    def generate(self):
        random_paths = random.sample(self.img_paths, self.n_shot * self.n_test)
        imgs = torch.stack([self.read_image_tensor(im_path) for im_path in random_paths])
        imgs = imgs.view(self.n_test, self.n_shot, *imgs.shape[1:])
        return imgs

    def read_image_tensor(self, im):
        img = Image.open(im).convert('RGB')
        img = self.tfms(img)
        return img

    def read_data(self, test_path):
        img_paths = glob(os.path.join(test_path, 'train', 'good', '*.png'))
        return img_paths
