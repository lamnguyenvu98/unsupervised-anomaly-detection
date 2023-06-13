from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch
import torch.nn.functional as F
from random import sample
from collections import OrderedDict
import numpy as np
import pickle
import os
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve

class PaDiM():
    def __init__(self, args) -> None:
        self.input_size = (args.MODEL.INPUT_SIZE, args.MODEL.INPUT_SIZE)
        self.device = args.TRAIN.DEVICE
        self.save_dir = args.TRAIN.SAVE_DIR
        self.checkpoint_path = args.INFERENCE.CHECKPOINT_PATH

        if args.TRAIN.PRETRAINED:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
        else:
            self.backbone = resnet18().to(self.device)
        
        self.t_d = 448
        # random pick features
        self.d = args.INFERENCE.REDUCE_FEATURES
        self.backbone.eval()
        self.idx = torch.tensor(sample(range(0, self.t_d), self.d)).to(self.device)
        
        self.output_layers = list()
        
        self.backbone.layer1[-1].register_forward_hook(self.hooks)
        self.backbone.layer2[-1].register_forward_hook(self.hooks)
        self.backbone.layer3[-1].register_forward_hook(self.hooks)
        
        self.distribution = None
        self.threshold = None
        self.max_score = None
        self.min_score = None

    def predict(self, data):
        '''
            data [torch.Tensor]: input data which has shape (batch, C, H, W)
                H, W    = (224, 224)
                C       = 3
                batch   = 1
        '''
        self.backbone.eval()
        
        outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        
        with torch.no_grad():
            _ = self.backbone(data.to(self.device))
        
        for k, v in zip(outputs.keys(), self.output_layers):
            outputs[k].append(v.detach())
        
        self.output_layers.clear()

        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, outputs[layer_name])
        
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        dist_list = []
        for i in range(H * W):
            mean = self.distribution[0][:, i]
            conv_inv = torch.linalg.inv(self.distribution[1][:, :, i])
            dist = [self.mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)
        
        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=data.size(2), mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        # self.max_score = score_map.max()
        # self.min_score = score_map.min()
        scores = (score_map - self.min_score) / (self.max_score - self.min_score)

        return scores

    def train(self, dataloader):
        self.backbone.eval()
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for img in dataloader:
            img = img.to(self.device)

            with torch.no_grad():
                _ = self.backbone(img)

            for k, v in zip(train_outputs.keys(), self.output_layers):
                train_outputs[k].append(v.detach())
            
            self.output_layers.clear()

        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0)
        cov = torch.zeros(C, C, H * W).to(self.device)
        I = torch.eye(C).to(self.device)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
        # save learned distribution
        self.distribution = [mean, cov]

    def evaluate(self, dataloader):
        self.backbone.eval()
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        gt_list = list()

        for img, label in dataloader:
            img = img.to(self.device)

            with torch.no_grad():
                _ = self.backbone(img)

            for k, v in zip(test_outputs.keys(), self.output_layers):
                test_outputs[k].append(v.detach())
            
            gt_list.extend(label.cpu().detach().numpy())
            
            self.output_layers.clear()

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        dist_list = []
        for i in range(H * W):
            mean = self.distribution[0][:, i]
            conv_inv = torch.linalg.inv(self.distribution[1][:, :, i])
            dist = [self.mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.input_size[0], mode='bilinear',
                                    align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        self.max_score = score_map.max()
        self.min_score = score_map.min()
        scores = (score_map - self.min_score) / (self.max_score - self.min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        # self.threshold = self.find_optimal_threshold(gt_list, img_scores)
        # print("Optimal threshold is:", self.threshold)
        return img_roc_auc

    def mahalanobis_torch(self, u, v, cov):
        delta = u - v
        m = torch.dot(delta, torch.matmul(cov, delta))
        return torch.sqrt(m)

    # def find_optimal_threshold(self, gt_list, img_scores):
    #     precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
    #     a = 2 * precision * recall
    #     b = precision + recall
    #     f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    #     threshold = thresholds[np.argmax(f1)]
    #     return threshold

    def save_checkpoint(self, filename):
        ckp = {
            "max_score": self.max_score,
            "min_score": self.min_score,
            "threshold": self.threshold,
            "idx": self.idx,
            "dist": self.distribution
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(ckp, path)
        # with open(path, 'wb') as f:
        #     pickle.dump(ckp, f)
    
    def load_checkpoint(self):
        # with open(self.checkpoint_path, 'rb') as f:
        #     ckp = pickle.load(f)
        ckp = torch.load(self.checkpoint_path, map_location=self.device)
        self.max_score = ckp['max_score']
        self.min_score = ckp['min_score']
        self.threshold = ckp['threshold']
        self.idx = ckp['idx']
        self.distribution = ckp['dist']
        
    def hooks(self, module, input, output):
        self.output_layers.append(output)

# helper functions
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z