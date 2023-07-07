from DFR.models.feature_extractor import FeatureExtractor
from DFR.models.encoder import AutoEncoder

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.decomposition import PCA
import re
from sklearn.metrics import roc_auc_score

from dotmap import DotMap

class DFR(object):
    def __init__(self, args: DotMap, latent_dim: int = 321):
        self.device = args.TRAIN.DEVICE
        self.input_size = (args.MODEL.INPUT_SIZE, args.MODEL.INPUT_SIZE)
        self.num_layers = args.TRAIN.NUM_LAYERS
        self.lr = args.TRAIN.LEARNING_RATE
        self.save_dir = args.TRAIN.SAVE_DIR
        self.checkpoint_path = args.TRAIN.CHECKPOINT_PATH
        self.latent_dim = latent_dim

        self.feature_extractor = FeatureExtractor(input_size=self.input_size, 
                                                  kernel_size=(4, 4),
                                                  num_layers=self.num_layers).to(self.device)
        
        # stop gradient in feature_extractor (feature_extractor isn't trained)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        
        self.feature_extractor.eval()

        self.auto_encoder = AutoEncoder(co = 5504,
                                       cd = self.latent_dim).to(self.device)

        self.threshold = None

        self.optimizer = torch.optim.Adam(self.auto_encoder.parameters(), lr=self.lr)

        self.prev_path = None
        self.best_top_k = dict()
        self.current_epoch = None

    def loss_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true)**2)
    
    def calculate_map(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true)**2, dim=1)

    def compute_threshold(self, dataloader: DataLoader, fpr: float = 0.005) -> None:
        error = []
        self.feature_extractor.eval()
        self.auto_encoder.eval()

        for X in tqdm(dataloader, desc='\tCompute Threshold'):
            with torch.no_grad():
                inp = self.feature_extractor(X.to(self.device))
                dec = self.auto_encoder(inp)
            score = self.calculate_map(dec, inp)
            error.append(score.detach().cpu().numpy())

        self.threshold = np.percentile(error, 100 - fpr)
    
    def compute_pca(self, dataloader: DataLoader) -> float:
        extract_per_sample = 20
        extractions = []

        self.feature_extractor.eval()
        pbar = tqdm(dataloader, desc='Compute PCA: ')
        for idx, X in enumerate(pbar):
            with torch.no_grad():
                out_features = self.feature_extractor(X.to(self.device))
            for feature in out_features:
                for _ in range(extract_per_sample):
                    row, col = random.randrange(feature.shape[1]), random.randrange(feature.shape[2])
                    extraction = feature[:, row, col]
                    extractions.append(extraction.detach().cpu().numpy())
        
        extractions = np.stack(extractions)
        print(f'Extractions shape: {extractions.shape}')
        pca = PCA(0.9, svd_solver="full")
        pca.fit(extractions)
        cd = pca.n_components_
        print(f"Components with explainable variance 0.9 -> {cd}")
        return cd
    
    def predict(self, data: torch.Tensor) -> np.ndarray:
        self.feature_extractor.eval()
        self.auto_encoder.eval()
        
        inp = self.feature_extractor(data.to(self.device))
        dec = self.auto_encoder(inp)
        
        scores = self.calculate_map(dec, inp)
        scores = F.interpolate(scores.unsqueeze(1), size=self.input_size, mode="bilinear", 
                               align_corners=True).squeeze(1).detach().numpy()
        return scores
        
    def train(self, dataloader: DataLoader) -> float:
        self.feature_extractor.eval()
        self.auto_encoder.train()
        losses = 0
        pbar = tqdm(dataloader, desc='\tTrain Phase: ')
        for x in pbar:
            inp = self.feature_extractor(x.to(self.device))
            dec = self.auto_encoder(inp)
            loss = self.loss_function(dec, inp)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            pbar.set_postfix_str(f"loss: {loss.item():.5f}")
        losses /= len(dataloader)
        return losses

    def evaluate(self, dataloader: DataLoader, threshold: float) -> tuple[float, float]:
        self.feature_extractor.eval()
        self.auto_encoder.eval()
        gt_list = list()
        pred_list = list()
        loss_eval = 0
        pbar = tqdm(dataloader, desc='\tTest Phase: ')
        for x, label in pbar:
            with torch.no_grad():
                inp = self.feature_extractor(x.to(self.device))
                dec = self.auto_encoder(inp)
                loss = self.loss_function(dec, inp)
            scores = self.calculate_map(dec, inp)
            scores = F.interpolate(scores.unsqueeze(1), size=self.input_size, mode="bilinear", 
                       align_corners=True).squeeze()
            loss_eval += loss.item()
            pbar.set_postfix_str(f"loss: {loss.item():.5f}")
            gt_list.extend(label.long().cpu().numpy().tolist())
            pred_list.append(scores.cpu().numpy())

        gt_list = np.array(gt_list)
        pred_list = np.concatenate(pred_list, axis=0)
        pred_list = pred_list.reshape(pred_list.shape[0], -1).max(axis=1)
        roc_auc = roc_auc_score(gt_list, pred_list)
        loss_eval /= len(dataloader)
        return roc_auc, loss_eval

    def save_checkpoint(self, save_path: str) -> None:
        ckp = {
            'best_top_k': self.best_top_k,
            'prev_path': self.prev_path,
            'threshold': self.threshold,
            'latent_dim': self.latent_dim,
            'autoencoder': self.auto_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.current_epoch:
            ckp.update({'current_epoch': self.current_epoch})
        torch.save(ckp, save_path)

    def save_last(self, current_epoch: int, filename: str) -> None:
        self.current_epoch = current_epoch
        path = os.path.join(self.save_dir, filename)
        if self.prev_path is None:
            self.save_checkpoint(path)
            self.prev_path = path
        else:
            if os.path.exists(self.prev_path):
                os.remove(f"{self.prev_path}")
            self.save_checkpoint(path)
            self.prev_path = path

    def format_checkpoint_name(self, metrics: dict, filename: str | None = None, prefix: str = "") -> str:
        file_name = "{epoch}" if filename is None else filename
        
        groups = re.findall(r"(\{.*?)[:\}]", file_name)
        if len(groups) > 0:
            for group in groups:
                name = group[1:]

                file_name = file_name.replace(group, name + "={" + name)

            if name not in metrics:
                metrics[name] = 0
            
            file_name = file_name.format(**metrics)
        
        if prefix:
            file_name = '-'.join([prefix, file_name])

        return file_name

    def save_top_k(self, metrics: dict, monitor: str, k: int = 1, filename: str | None = None) -> None:
        assert "epoch" in metrics.keys(), f"Add epoch into {metrics}"
        assert monitor in metrics.keys(), f"{monitor} is not existed in {metrics}. Select right monitor value."
        self.current_epoch = metrics['epoch']
        filename = self.format_checkpoint_name(metrics, filename)
        path = os.path.join(self.save_dir, filename)
        if len(self.best_top_k) < k and path not in self.best_top_k:
            self.best_top_k[path] = metrics[monitor]
        else:
            smallest_k = min(self.best_top_k.values(), key=lambda x: x)
            if smallest_k <= metrics[monitor]:
                smallest_path_k = [k for k, v in self.best_top_k.items() if v == smallest_k][0]
                if os.path.exists(smallest_path_k):
                    os.remove(f"{smallest_path_k}")
                del self.best_top_k[smallest_path_k]
                self.best_top_k[path] = metrics[monitor]
        
        for k_path, k_vals in self.best_top_k.items():
            if os.path.exists(k_path):
                continue
            self.save_checkpoint(k_path)

    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is None:
            ckp = torch.load(self.checkpoint_path, map_location=self.device)
        else:
            ckp = torch.load(checkpoint_path, map_location=self.device)
        self.best_top_k = ckp['best_top_k']
        self.prev_path = ckp['prev_path']
        self.threshold = ckp['threshold']
        self.latent_dim = ckp['latent_dim']
        self.auto_encoder = AutoEncoder(
            co = 5504,
            cd = self.latent_dim).to(self.device)
        self.auto_encoder.load_state_dict(ckp['autoencoder'])
        self.optimizer.load_state_dict(ckp['optimizer'])
        self.current_epoch = ckp['current_epoch']
        print("[INFO] Load model successful!")