import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

transform = lambda img,dims:transforms.Compose([
	transforms.Resize(dims),
	transforms.ToTensor(),
	lambda x:x[:3],
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])(img)

class GramMatrix(nn.Module):
	def forward(self,x):
		_,d,h,w = x.shape
		x = x.view(d,h*w)
		return torch.mm(x,x.T)

def read_img(f,dims,transform=transform):
	img = Image.open(f)
	img = transform(img,dims)
	img = img.unsqueeze(0)
	return img

def to_img(t):
	img = (t.squeeze().permute(1,2,0).detach().numpy()*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]))
	img = (img * 255).astype(np.uint8)
	return img

def save_img(t,f):
	img = to_img(t)
	img = Image.fromarray(img)
	#print(img.shape)
	img.save(f)