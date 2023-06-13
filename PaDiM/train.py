from torchvision import transforms
from datasets import TrainDataset, TestDataset
from padim import PaDiM
from load_config import read_args
from torch.utils.data import DataLoader
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

model = PaDiM(args)

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

train_dl = DataLoader(train_set, batch_size=args.TRAIN.BATCH_SIZE, num_workers=os.cpu_count(), 
                      shuffle=True, pin_memory=True, drop_last=False)

test_dl = DataLoader(test_set, batch_size=args.TRAIN.BATCH_SIZE, num_workers=os.cpu_count(), 
                     shuffle=False, pin_memory=True)

print('Size of training set: ', len(train_set))
print('Size of testing set: ', len(test_set))

# train
model.train(train_dl)

# test
roc_auc = model.evaluate(test_dl)

print("ROC AUC score:", roc_auc)
print("Threshold: ", model.threshold)
model.save_checkpoint(filename=f'train_transistor_{roc_auc:.2f}.pt')