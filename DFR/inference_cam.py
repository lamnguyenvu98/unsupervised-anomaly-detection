from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[0].absolute()))

import torch
from torchvision import transforms
import cv2
import numpy as np
from models.dfr import DFR

from load_config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--camidx', '-i', default=0, type=int, help="Camera index")
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    data = cv2.resize(frame, (224, 224))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = torch.tensor(data.copy()/255.).permute(2, 0, 1).unsqueeze(0).float()
    data = tfms(data)
    scores = model.predict(data)
    # 0: normal, 1: anomaly
    prediction = np.any(scores > model.threshold).astype('int')
    # print('Prediction: ', classes[prediction])
    
    # Get anomaly mask
    mask = np.where(scores.squeeze(0) > model.threshold, 1, 0)
    mask = cv2.resize((mask[..., None] * 255).astype(np.uint8), (w, h))

    # visualize anomaly region
    colormap = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
    anomaly_frame = cv2.addWeighted(colormap, 0.7, frame, 0.7, 0)
    cv2.putText(anomaly_frame, 'Prediction: ' + classes[prediction], (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.imshow("frame", anomaly_frame)
    if cv2.waitKey(1) & 0xFF == 27: # Esc
        break

cap.release()
cv2.destroyAllWindows()
    
