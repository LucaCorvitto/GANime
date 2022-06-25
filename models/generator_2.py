import torch.nn as nn

nz = 128

class GenResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(GenResidualBlock, self).__init__()
        p = kernel_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p),
            nn.BatchNorm2d(out_channels, 0.8),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None        
    
    def forward(self, x):
        identity = x
        output = self.conv1(x)
        output = self.conv2(output)
        identity = identity if self.proj is None else self.proj(identity)
        output = output + identity
        return output

class Generator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, out_channel=3):
        super(Generator, self).__init__()
        self.fc = nn.Linear(nz, 1024*4*4)
        self.G = nn.Sequential(
            GenResidualBlock(1024, 512),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 512, 8, 8)
            GenResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 256, 16, 16)
            GenResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 128, 32, 32)
            GenResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 64, 64, 64)
            GenResidualBlock(64, 64),
            #nn.Conv2d(64, out_channel, 3, padding=1), # (N, 3, 64, 64)
            nn.Conv2d(64, out_channel, 9, padding=4), # (N, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, x):
        batch = x.size(0)
        h = self.fc(x)
        h = h.view(batch, 1024, 4, 4)
        out = self.G(h)
        return out