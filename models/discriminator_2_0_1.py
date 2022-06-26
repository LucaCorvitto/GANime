import torch.nn as nn

nz = 128  # BatchNormalization

class DiscResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DiscResidualBlock, self).__init__()
        p = kernel_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p),
            nn.BatchNorm2d(out_channels, 0.8)
        ) 
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None 
        self.lr = nn.LeakyReLU(0.2)       
    
    def forward(self, x):
        identity = x
        output = self.conv1(x)
        output = self.conv2(output)
        identity = identity if self.proj is None else self.proj(identity)
        output = output + identity
        output = self.lr(output)
        return output

class Discriminator(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, base=32, in_channel=3):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(in_channel, base, 4, 2, padding=1), #32x32
            nn.BatchNorm2d(base, 0.8),
            nn.LeakyReLU(0.2),
            DiscResidualBlock(base, base),
            DiscResidualBlock(base, base),
            nn.Conv2d(base, base*2, 4, 2, padding=1),  #16x16
            nn.BatchNorm2d(base*2, 0.8),
            nn.LeakyReLU(0.2),
            DiscResidualBlock(base*2, base*2),
            DiscResidualBlock(base*2, base*2),
            nn.Conv2d(base*2, base*4, 4, 2, padding=1), #8x8
            nn.BatchNorm2d(base*4, 0.8),
            nn.LeakyReLU(0.2),
            DiscResidualBlock(base*4, base*4),
            DiscResidualBlock(base*4, base*4),
            nn.Conv2d(base*4, base*8, 4, 2, padding=1), #4x4
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(4096, 1) # (N, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        y = self.sig(y)
        return y