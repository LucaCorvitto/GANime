import torch
import torch.nn as nn


class Discriminator(nn.Module):
	def __init__(self, l=0.2):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(l),

			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(l),

			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(l),
			nn.Conv2d(1024, 1, kernel_size=1)
		)

	def forward(self, x): 
		#print ('D input size :' +  str(x.size()))
		y = self.net(x)
		#print ('D output size :' +  str(y.size()))
		si = torch.sigmoid(y).view(y.size()[0])
		#print ('D output : ' + str(si))
		return si