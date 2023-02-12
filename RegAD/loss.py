import torch.nn.functional as F

def negative_cosine_similairty(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=1).mean()