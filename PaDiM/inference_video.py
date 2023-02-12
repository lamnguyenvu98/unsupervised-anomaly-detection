import cv2
import torch
import numpy as np

from padim import PaDiM

from load_config import read_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

padim = PaDiM(args)

padim.load_checkpoint()

classes = ['OK', 'NG']

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    h, w = img.shape[:2]
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = cv2.resize(img, (224, 224))
    data = torch.tensor(data/255.).permute(2, 0, 1).unsqueeze(0).float()
    scores = padim.predict(data)
    img_scores = np.any(scores > 0.5).astype('int')
    print("prediction: ", classes[img_scores])

    mask = np.where(scores.squeeze() > 0.5, 1, 0)
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))

    colormap = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)
    anomaly_frame = cv2.addWeighted(colormap, 0.7, img, 0.7, 0)
    cv2.putText(anomaly_frame, 'Prediction: ' + classes[img_scores], (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.imshow("frame", anomaly_frame)
    if cv2.waitKey(1) & 0xFF == 27: # Esc to escape
        break

cap.release()
cv2.destroyAllWindows()
