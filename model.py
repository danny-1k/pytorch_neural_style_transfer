import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
	def __init__(self):
		super().__init__()
		vgg_features = models.vgg16(pretrained=True).features
		for i in vgg_features.parameters():
			i.requires_grad_(False)
		self.vgg_features = vgg_features
		self.use_layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','17':'conv4_1','19':'conv4_2','24':'conv5_1'}
	def forward(self,x):
		features = {}
		for idx,module in self.vgg_features._modules.items():
			x = module(x)
			if str(idx) in self.use_layers:
				features[self.use_layers[str(idx)]] = x

		return features


