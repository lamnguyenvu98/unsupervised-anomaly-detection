from RegAD.datasets import TrainDataset, TestDataset, TaskTrainSampler, GenerateSupportSet
from RegAD.models import RegAD
import argparse
import os
from load_config import read_args
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, type=str, help="path to yaml config file")
ar = parser.parse_args()

args = read_args(ar.config)

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

train_set = TrainDataset(root=args.TRAIN.TRAIN_DATA_DIR, ignore_class=['transistor'], tfms=tfms)
test_set = TestDataset(test_path=args.TRAIN.TEST_DATA_DIR, tfms=tfms)

train_sampler = TaskTrainSampler(train_set, n_shot=args.TRAIN.N_SHOT, 
                                 batch_size=args.TRAIN.BATCH_SIZE)

train_dl = DataLoader(train_set, batch_sampler=train_sampler, num_workers=os.cpu_count(),
                     collate_fn=train_sampler.episodic_collate_fn, pin_memory=True)

test_dl = DataLoader(test_set, batch_size=32, num_workers=os.cpu_count(),
                    pin_memory=True)

support_set_eval = GenerateSupportSet(
                                  test_path=args.TRAIN.TEST_DATA_DIR,
                                  num_test=args.TRAIN.N_TEST, 
                                  n_shot=args.TRAIN.N_SHOT).generate()

print('Size of train set:', len(train_set))
print('Size of test set:', len(test_set))
print('Number of testing rounds:', len(support_set_eval))
print('Size of support set:', support_set_eval.size(1))

model = RegAD(args)

if args.TRAIN.RESUME_TRAIN:
    model.load_checkpoint()
    start_epoch = model.current_epoch + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, args.TRAIN.EPOCH):
    print(f"EPOCH: {epoch+1}/{args.TRAIN.EPOCH}")
    model.adjust_learning_rate(epoch, args.TRAIN.EPOCH)
    train_loss = model.train(train_dl)
    # Evaluate model after 2 epoch
    if epoch >= args.TRAIN.BURN_IN:
        auroc_score, best_score, val_loss = model.evaluate(test_dl, support_set_eval)
        # save best k model for average auroc_score
        metrics = {'epoch': epoch, 'auroc_score': auroc_score}
        model.save_top_k(metrics, monitor='auroc_score', filename="model-{epoch:02d}-{auroc_score:.2f}.pt", k=3)
        print(f"[RESULT] Train Loss: {train_loss:.5f} , Val Loss: {val_loss:.5f} , AUROC: {auroc_score:.2f} - BEST: {best_score:.2f}\n")
    else:
        print(f"[RESULT] Train Loss: {train_loss:.5f}\n")
    # save latest checkpoint
    model.save_last(epoch=epoch, filename=f"last.pt")
