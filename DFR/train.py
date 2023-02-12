import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import os

from dataset import TestDataset, TrainDataset
from models.dfr import DFR
from load_config import read_args

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

tfms = transforms.Compose(
            [
                transforms.Resize(args.MODEL.INPUT_SIZE, args.MODEL.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
)

train_set = TrainDataset(root=args.TRAIN.DATA_DIR, tfms=tfms)
test_set = TestDataset(root=args.TRAIN.DATA_DIR, tfms=tfms)

train_dl = DataLoader(train_set, batch_size=args.TRAIN.BATCH_SIZE, 
                      num_workers=os.cpu_count(), shuffle=True, pin_memory=True, drop_last=True)

test_dl = DataLoader(test_set, batch_size=args.TRAIN.BATCH_SIZE, 
                     num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

print('Size of training set: ', len(train_set))
print('Size of test set: ', len(test_set))

dfr = DFR(args)

for epoch in range(0, args.TRAIN.EPOCH):
    loss_train = dfr.train(train_dl)
    print(f'EPOCH {epoch+1} - Train Loss: {loss_train}')

print('[INFO] Train Successful!')
print('[INFO] TESTING')

dfr.compute_threshold(train_dl, fpr=0.005)
roc_auc_avg, loss_eval = dfr.evaluate(test_dl, dfr.threshold)

print("ROC AUC score:",roc_auc_avg)

print('Save checkpoint')
dfr.save_checkpoints(filename='best.pt')