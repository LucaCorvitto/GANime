import torch.nn as nn

nz = 128
#base = 64

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

class PixelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()

    def forward(self, tensor):
        output = self.conv(tensor)
        output = self.pixel_shuffle(output)
        output = self.bn(output)
        output = self.relu(output)
        return output

class Generator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, base=64, out_channel=3):
        super(Generator, self).__init__()
        self.base = base
        self.fc = nn.Linear(nz, base*16*16)
        self.b1 = nn.Sequential(
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2)
        )
        self.resBlocks = nn.Sequential(
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            GenResidualBlock(base, base),
            nn.BatchNorm2d(base),
            nn.PReLU()
        )
        self.ending = nn.Sequential(
            PixelBlock(base, base*4),
            PixelBlock(base, base*4),
            #nn.Conv2d(64, out_channel, 3, padding=1), # (N, 3, 64, 64)
            nn.Conv2d(base, out_channel, 9, padding=4), # (N, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, x):
        batch = x.size(0)
        h = self.fc(x)
        h = h.view(batch, self.base, 16, 16)
        h = self.b1(h)
        r_tensor = h
        out = self.resBlocks(h)
        out += r_tensor          #elementwise sum of residual block
        out = self.ending(out)
        return out