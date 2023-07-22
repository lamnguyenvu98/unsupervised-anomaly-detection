import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import os

import argparse

from load_config import read_args

from DFR.datasets import TrainDataset, TestDataset
from DFR.models import DFR

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

tfms = transforms.Compose(
            [
                transforms.Resize([args.MODEL.INPUT_SIZE, args.MODEL.INPUT_SIZE]),
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

latent_dim = DFR(args).compute_pca(train_dl)

model = DFR(args, latent_dim=latent_dim)

if args.TRAIN.RESUME_TRAIN:
    model.load_checkpoint()
    start_epoch = model.current_epoch + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, args.TRAIN.EPOCH):
    print(f"Epoch {epoch} / {args.TRAIN.EPOCH}")
    loss_train = model.train(train_dl)
    # Evaluate and save checkpoint k best checkpoint after 100 epoch
    if epoch >= args.TRAIN.BURN_IN:
        model.compute_threshold(train_dl, fpr=0.005)
        auroc_score, loss_eval = model.evaluate(test_dl, model.threshold)
        metrics = {'epoch': epoch, 'auroc_score': auroc_score}
        model.save_top_k(metrics, monitor='auroc_score', k=3, filename='model-{epoch:03d}-{auroc_score:.2f}.pt')
        print(f'[RESULT] Train Loss: {loss_train:.5f} - Val Loss: {loss_eval:.5f} - AUROC: {auroc_score:.3f}\n')
    else:
        print(f'[RESULT] Train Loss: {loss_train:.5f}\n')
    # Always save last state of the model
    model.save_last(epoch, filename='last.pt')
