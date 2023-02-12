import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, kernel_size, output_size=None, dilation=1, stride=1):
        super(AvgFeatAGG2d, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        output = self.unfold(input)  # (b, cxkxk, h*w)
        output = torch.reshape(output, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
        # print(output.shape)
        output = torch.mean(output, dim=2)
        # output = self.fold(input)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self, input_size = (256, 256), kernel_size=(4, 4), stride = (4, 4), num_layers=16):
        super().__init__()
        assert num_layers <= 16, 'There are only 16 layers'
        self.num_layers = num_layers
        self.stride = stride
        self.out_features = []
        self.backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        for m in self.backbone.children():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(self.hook)
        
        self.up_sample = nn.Upsample(size=input_size, align_corners=None, mode='nearest')

        # feature processing
        padding_h = (kernel_size[0] - self.stride[0]) // 2
        padding_w = (kernel_size[1] - self.stride[1]) // 2
        self.padding = (padding_h, padding_w)
        self.replicationpad = nn.ReplicationPad2d((padding_w, padding_w, padding_h, padding_h))

        self.out_h = int((input_size[0] + 2 * self.padding[0] - 1 * (kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.out_w = int((input_size[1] + 2 * self.padding[1] - 1 * (kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        self.feature_agg = AvgFeatAGG2d(kernel_size=kernel_size, output_size=(self.out_h, self.out_w), 
                                        dilation=1, stride=self.stride)
        # self.avg_pool = nn.AvgPool2d(kernel_size=(4, 4), stride=self.stride)
    
    def forward(self, x):
        _ = self.backbone(x)
        out_features = torch.cat([self.feature_agg(self.replicationpad(self.up_sample(feature))) \
                                  for feature in self.out_features[:self.num_layers]], dim=1)
        self.out_features.clear()
        return out_features.view(out_features.size(0), out_features.size(1), self.out_h, self.out_w)
    
    def hook(self, module, input, output):
        self.out_features.append(output.detach())