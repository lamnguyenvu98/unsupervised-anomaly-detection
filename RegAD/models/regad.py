import os

from RegAD.models.stn import stn_net
from RegAD.models.siamese import Encoder, Predictor
from RegAD.utils.meter import AverageMeter
from RegAD.loss import negative_cosine_similairty
from RegAD.utils.funcs import *

from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import re

import math
from torch.utils.data import DataLoader

from dotmap import DotMap

class RegAD():
    def __init__(self, args: DotMap)-> None:
        self.device = args.TRAIN.DEVICE
        self.STN = stn_net(resnet_type=args.MODEL.BACKBONE, 
                           stn_mode=args.MODEL.STN_MODE, 
                           pretrained=args.TRAIN.PRETRAINED).to(self.device)
        self.ENC = Encoder().to(self.device)
        self.PRED = Predictor().to(self.device)
        
        self.num_test = args.TRAIN.N_TEST
        self.n_shot = args.TRAIN.N_SHOT
        self.save_dir = args.TRAIN.SAVE_DIR
        self.checkpoint_path = args.TRAIN.CHECKPOINT_PATH
        
        self.lr_stn = args.TRAIN.LEARNING_RATE_STN
        self.lr_enc = args.TRAIN.LEARNING_RATE_ENC
        self.lr_pred = args.TRAIN.LEARNING_RATE_PRED

        self.STN_optimizer = torch.optim.SGD(self.STN.parameters(), 
                                        lr=self.lr_stn, 
                                        momentum=args.TRAIN.MOMENTUM_STN)
        self.ENC_optimizer = torch.optim.SGD(self.ENC.parameters(), 
                                        lr=self.lr_enc, 
                                        momentum=args.TRAIN.MOMENTUM_ENC)
        self.PRED_optimizer = torch.optim.SGD(self.PRED.parameters(), 
                                         lr=self.lr_pred, 
                                         momentum=args.TRAIN.MOMENTUM_PRED)
        
        self.max_score = None
        self.min_score = None
        self.avg_max_score = None
        self.avg_min_score = None
        self.prev_path = None
        self.best_top_k = dict()
        self.curent_epoch = None

    def predict(self, query_image: torch.Tensor, support_distribution: torch.Tensor, support_feat: torch.Tensor, norm: str = 'avg') -> float:
        '''
            support_set [torch.Tensor]: shape (K, C, H, W)
                K: number of shot (number of images in support set)
            query_image [torch.Tensor]: shape (B, C, H, W)
                B: batch size
            
            norm ['avg' or 'best']: select average min/max score to normalize predicted score
                or best min/max score to normalize predicted score
        '''
        assert norm in ['avg', 'best'], '[ERROR] Invalid value. There are two options: "avg" and "best"'
        self.STN.eval()
        self.ENC.eval()
        self.PRED.eval()
                
        with torch.no_grad():
            query_feat = self.STN(query_image.to(self.device))
            z1 = self.ENC(query_feat)
            z2 = self.ENC(support_feat)
            p1 = self.PRED(z1)
            p2 = self.PRED(z2)
        
        stn_layers = {"layer1": self.STN.stn1_output, "layer2": self.STN.stn2_output, "layer3": self.STN.stn3_output}
        
        # Embedding concat
        embedding_vectors = stn_layers['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, stn_layers[layer_name], self.device)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        dist_list = []

        for i in range(H * W):
            mean = support_distribution[0][:, i]
            conv_inv = torch.linalg.inv(support_distribution[1][:, :, i])
            dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=query_image.size(2), mode='bilinear',
                                align_corners=True).squeeze(1).numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        scores = np.asarray(score_map)

        # Normalization
        # max_anomaly_score = scores.max()
        # min_anomaly_score = scores.min()
        if norm == 'best':
            scores = (scores - self.min_score) / (self.max_score - self.min_score)
        elif norm == 'avg':
            scores = (scores - self.avg_min_score) / (self.avg_max_score - self.avg_min_score)

        return scores
        
    def train(self, dataloader: DataLoader) -> float:
        self.STN.train()
        self.ENC.train()
        self.PRED.train()
        
        total_losses = AverageMeter()

        pbar = tqdm(dataloader, desc = '\tTrain Phase: ')
        
        for query_img, support_img in pbar:
            query_img, support_img = query_img.to(self.device), support_img.to(self.device)
            self.STN_optimizer.zero_grad()
            self.ENC_optimizer.zero_grad()
            self.PRED_optimizer.zero_grad()

            query_feature = self.STN(query_img.to(self.device))
            B,K,C,H,W = support_img.shape
            support_feature = self.STN(support_img.view(B*K, C, H, W))

            support_feature = support_feature / K
            support_feature = support_feature.view(B, K, *support_feature.shape[1:])
            support_feature = support_feature.sum(dim=1)

            z1 = self.ENC(query_feature)
            z2 = self.ENC(support_feature)
            p1 = self.PRED(z1)
            p2 = self.PRED(z2)
            
            total_loss = negative_cosine_similairty(p1, z2)/2 + negative_cosine_similairty(p2,z1)/2
            
            total_losses.update(total_loss.item(), query_img.size(0))
            total_loss.backward()
            
            self.STN_optimizer.step()
            self.ENC_optimizer.step()
            self.PRED_optimizer.step()

            pbar.set_postfix_str(f"Loss: {total_loss:.5f}")
        
        return total_losses.avg
    
    def evaluate(self, test_dataloader: DataLoader, support_set_eval: torch.Tensor) -> tuple[float, float, float]:
        self.STN.eval()
        self.ENC.eval()
        self.PRED.eval()

        meter = AverageMeter()
        
        n_test = len(support_set_eval)
        
        roc_auc_list = list()
        max_score_list = []
        min_score_list = []

        best_score = 0

        for i in range(n_test):
            query_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

            support_distribution, support_feat = self.calculate_distribution_support_set(support_set_eval[i, ...])
            
            gt_list = []
            pbar = tqdm(test_dataloader, desc=f'\tTest Phase {i}/{n_test}')
            for idx, (query_img, labels) in enumerate(pbar):
                #### Test phase
                with torch.no_grad():
                    query_feat = self.STN(query_img.to(self.device))
                    z1 = self.ENC(query_feat)
                    z2 = self.ENC(support_feat)
                    p1 = self.PRED(z1)
                    p2 = self.PRED(z2)

                loss = negative_cosine_similairty(p1, z2)/2 + negative_cosine_similairty(p2, z1)/2

                meter.update(loss.item(), query_img.size(0))

                query_outputs['layer1'].append(self.STN.stn1_output)
                query_outputs['layer2'].append(self.STN.stn2_output)
                query_outputs['layer3'].append(self.STN.stn3_output)

                gt_list.extend(labels.numpy().tolist())

            for k, v in query_outputs.items():
                query_outputs[k] = torch.cat(v, 0)

            # Embedding concat for all query images in test set
            embedding_vectors = query_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, query_outputs[layer_name], self.device)

            # calculate distance matrix
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            dist_list = []

            for i in range(H * W):
                mean = support_distribution[0][:, i]
                conv_inv = torch.linalg.inv(support_distribution[1][:, :, i])
                dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
                dist_list.append(dist)

            dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

            # upsample
            score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()

            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)
            
            scores = np.asarray(score_map)

            # Normalization
            max_score = scores.max()
            min_score = scores.min()
            scores = (scores - min_score) / (max_score - min_score)

            # calculate image-level ROC AUC score
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            gt_list = np.array(gt_list)
            img_roc_auc = roc_auc_score(gt_list, img_scores)

            if best_score < img_roc_auc:
                best_score = img_roc_auc
                self.max_score = max_score
                self.min_score = min_score

            roc_auc_list.append(img_roc_auc)
            max_score_list.append(max_score)
            min_score_list.append(min_score)
        
        roc_auc_avg = np.mean(roc_auc_list)
        self.avg_max_score = np.mean(max_score_list)
        self.avg_min_score = np.mean(min_score_list)
        
        return roc_auc_avg, best_score, meter.avg
    
    def augment_support_set(self, support_img: torch.Tensor) -> torch.Tensor:
        augment_support_img = support_img
        # rotate img with small angle
        for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
            rotate_img = rot_img(support_img, angle)
            augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
        # translate img
        for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(support_img, a, b)
            augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
        # hflip img
        flipped_img = hflip_img(support_img)
        augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
        # rgb to grey img
        greyed_img = grey_img(support_img)
        augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
        # rotate img in 90 degree
        for angle in [1,2,3]:
            rotate90_img = rot90_img(support_img, angle)
            augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
        
        augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]

        return augment_support_img
    
    def calculate_distribution_support_set(self, support_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        support_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        augment_support_img = self.augment_support_set(support_img)
        with torch.no_grad():
            support_feat = self.STN(augment_support_img.to(self.device))
        support_feat = torch.mean(support_feat, dim=0, keepdim=True)

        support_outputs['layer1'].append(self.STN.stn1_output)
        support_outputs['layer2'].append(self.STN.stn2_output)
        support_outputs['layer3'].append(self.STN.stn3_output)

        for k, v in support_outputs.items():
            support_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = support_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, support_outputs[layer_name], self.device)

        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0) # (C, H * W)
        cov = torch.zeros(C, C, H * W).to(self.device)
        I = torch.eye(C).to(self.device)
        for i in range(H * W):
            cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
        
        support_outputs = [mean, cov]

        return support_outputs, support_feat

    def adjust_learning_rate(self, epoch: int, end_epoch: int) -> None:
        """Decay the learning rate based on schedule"""

        optimizers = [self.STN_optimizer, self.ENC_optimizer, self.PRED_optimizer]
        init_lrs = [self.lr_stn, self.lr_enc, self.lr_pred]

        for i in range(3):
            cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / end_epoch))
            for param_group in optimizers[i].param_groups:
                param_group['lr'] = cur_lr

    def save_checkpoint(self, path):
        ckp = {
            "STN": self.STN.state_dict(),
            "ENC": self.ENC.state_dict(),
            "PRED": self.PRED.state_dict(),
            "STN_optimizer": self.STN_optimizer.state_dict(),
            "ENC_optimizer": self.ENC_optimizer.state_dict(),
            "PRED_optimizer": self.PRED_optimizer.state_dict(),
            "max_score": self.max_score,
            "min_score": self.min_score,
            "avg_max_score": self.avg_max_score,
            "avg_min_score": self.avg_min_score,
            "best_top_k": self.best_top_k,
            "prev_path": self.prev_path
        }
        if self.curent_epoch:
            ckp.update({'curent_epoch': self.curent_epoch})
        torch.save(ckp, path)

    def save_last(self, epoch: int, filename: str) -> None:
        self.curent_epoch = epoch
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
        self.curent_epoch = metrics['epoch']
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
        self.STN.load_state_dict(ckp['STN'])
        self.ENC.load_state_dict(ckp['ENC'])
        self.PRED.load_state_dict(ckp['PRED'])
        self.STN_optimizer.load_state_dict(ckp['STN_optimizer'])
        self.ENC_optimizer.load_state_dict(ckp['ENC_optimizer'])
        self.PRED_optimizer.load_state_dict(ckp['PRED_optimizer'])
        self.max_score = ckp['max_score']
        self.min_score = ckp['min_score']
        self.avg_max_score = ckp["avg_max_score"]
        self.avg_min_score = ckp["avg_min_score"]
        self.best_top_k = ckp['best_top_k']
        self.prev_path = ckp['prev_path']
        self.curent_epoch = ckp['curent_epoch']
        print("[INFO] Load model weight successful!")
