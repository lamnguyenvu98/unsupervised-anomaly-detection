from models.regad import RegAD
import torch
import cv2
import numpy as np
import os
from torchvision import transforms
import glob
from utils.load_config import read_args
from utils.funcs import read_support_set
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--image", '-i', required=True, type=str, help="Path to input image")
parser.add_argument("--threshold", '-t', default=0.5, type=float, help="Threshold number for prediction")
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

classes = ['OK', 'NG']

args = read_args(ar.config)

model = RegAD(args)

model.load_checkpoint()

# read support images
support_set = read_support_set(args.INFERENCE.SUPPORT_SET_PATH)

support_distribution, support_feat = model.calculate_distribution_support_set(support_set)

img = cv2.imread(ar.image)
w, h = img.shape[:2]
data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = cv2.resize(img, (224, 224))
data = torch.tensor(data/255).permute(2, 0, 1).unsqueeze(0).float()

scores = model.predict(data, support_distribution, support_feat)

img_scores = np.any(scores > ar.threshold).astype('int')

print("Prediction: ", classes[img_scores])

mask = np.where(scores.squeeze() > ar.threshold, 1, 0)
mask = (mask * 255).astype(np.uint8)
mask = cv2.resize(mask, (w, h))

colormap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
anomaly_frame = cv2.addWeighted(colormap, 0.7, img, 0.5, 0)

plt.imshow(anomaly_frame)
plt.show()