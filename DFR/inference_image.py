from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[0].absolute()))

import torch
from torchvision import transforms
import cv2
import numpy as np
from models.dfr import DFR
import matplotlib.pyplot as plt
from load_config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

model = DFR(args)
model.load_checkpoint()

classes = ['OK', 'NG']

tfms = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

frame = cv2.imread('/home/pep/Drive/PCLOUD/Projects/Anomaly-Detection/DFR/003.png')
h, w = frame.shape[:2]
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (256, 256))
data = torch.tensor(frame/255.).permute(2, 0, 1).unsqueeze(0).float()
data = tfms(data)
scores = model.predict(data)

# 0: normal, 1: anomaly
prediction = torch.any((scores > model.threshold).view(-1), dim=-1).int().item()
print('Prediction: ', classes[prediction])

mask_ng = scores.detach().cpu().numpy().squeeze()
# mask_ng = cv2.resize(mask_ng, (256, 256))
mask_ng = np.where(mask_ng > model.threshold, 1, 0)
colormap = cv2.applyColorMap((mask_ng * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
anomaly_frame = cv2.addWeighted(colormap, 0.7, frame, 0.7, 0)
plt.imshow(anomaly_frame)
plt.show()
plt.imshow(mask_ng, cmap='gray')
plt.show()