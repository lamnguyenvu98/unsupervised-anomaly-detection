from datasets.one_class_dataset import TrainDataset, TestDataset, TaskTrainSampler
from datasets.generate_sup import GenerateSupportSet
from models.regad import RegAD
import argparse
import os
from utils.load_config import read_args
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

model = RegAD(args)

tfms = transforms.Compose(
            [
                transforms.Resize([args.MODEL.INPUT_SIZE, args.MODEL.INPUT_SIZE]),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # )
            ]
)

train_set = TrainDataset(
    root=args.TRAIN.DATA_DIR,
    tfms=tfms
)

train_sampler = TaskTrainSampler(
    dataset=train_set,
    n_shot=args.TRAIN.N_SHOT,
    batch_size=args.TRAIN.BATCH_SIZE
)

train_dl = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=os.cpu_count(),
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

test_set = TestDataset(
    root=args.TRAIN.DATA_DIR,
    tfms=tfms
)

test_dl = DataLoader(
    test_set,
    batch_size=args.TRAIN.BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True
)

support_set_eval = GenerateSupportSet(root=args.TRAIN.DATA_DIR,
                                  num_test=args.TRAIN.N_TEST, 
                                  n_shot=args.TRAIN.N_SHOT).generate()

best_roc = 0

for epoch in range(args.TRAIN.EPOCH):
    train_loss = model.train(train_dl)
    roc_avg, val_loss = model.evaluate(test_dl, support_set_eval)
    print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.5f} , Val Loss: {val_loss:.5f} , ROC AUC {args.TRAIN.N_TEST} rounds : {roc_avg:.3f}")
    if best_roc < roc_avg:
        model.save_checkpoint(filename=f"{epoch+1}_{args.TRAIN.N_SHOT}_{roc_avg:.2f}.pt")
        best_roc = roc_avg