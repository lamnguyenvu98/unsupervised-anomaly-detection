import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from padim import PaDiM
from load_config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

padim = PaDiM(args)

padim.load_checkpoint()

img = cv2.imread('/home/pep/Drive/PCLOUD/Projects/Anomaly-Detection/DFR/001.png')
h, w = img.shape[:2]
data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = cv2.resize(img, (256, 256))
data = torch.tensor(data/255.).permute(2, 0, 1).unsqueeze(0).float()
scores = padim.predict(data)
img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
print(img_scores)

mask = np.where(scores.squeeze() > padim.threshold, 1, 0)
mask = (mask * 255).astype(np.uint8)
mask = cv2.resize(mask, (w, h))

colormap = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
anomaly_frame = cv2.addWeighted(colormap, 0.5, img, 0.5, 0)

plt.imshow(anomaly_frame)
plt.show()