import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, co, cd) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=co, out_channels=(co + cd)//2, kernel_size=(1, 1)),
            nn.BatchNorm2d((co + cd)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(co + cd)//2, out_channels=2*cd, kernel_size=(1, 1)),
            nn.BatchNorm2d(2*cd),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*cd, out_channels=cd, kernel_size=(1, 1)),
            nn.BatchNorm2d(cd),
            nn.ReLU(),
            nn.Conv2d(in_channels=cd, out_channels=2*cd, kernel_size=(1, 1)),
            nn.BatchNorm2d(2*cd),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*cd, out_channels=(co + cd)//2, kernel_size=(1, 1)),
            nn.BatchNorm2d((co + cd)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=(co + cd)//2, out_channels=co, kernel_size=(1, 1)),
            nn.BatchNorm2d(co),
        )
    
    def forward(self, fx):
        return self.features(fx)