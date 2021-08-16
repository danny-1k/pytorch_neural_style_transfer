import torch
import torch.optim as optim
import torchvision
import utils
from model import VGG
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Neural Style Transfer")
parser.add_argument('--content', help="Content image", default = "imgs/content/memes.jpg")
parser.add_argument('--style', help="Style image", default = "imgs/style/edtaonisl.jpg")
parser.add_argument('--alpha', help="Content weight",type=float, default = 1.0)
parser.add_argument('--beta', help="Style_weight",type=float, default = 1.0)
parser.add_argument('--gpu', help="GPU enabled?",type=bool, default = False)
parser.add_argument('--output_size', help="Output size",type=str, default = (200,200))
parser.add_argument('--output_file', help="Output dest", default = "result.jpg")
args = parser.parse_args()
print(args.output_size)
args.output_size = eval(args.output_size)
content = utils.read_img(args.content,args.output_size)
style = utils.read_img(args.style,args.output_size)

target = content.clone().requires_grad_(True)

alpha = args.alpha
beta = args.beta

device = 'cuda' if args.gpu else 'cpu'
vgg = VGG()

vgg.to(device)
content.to(device)
style.to(device)

style_weights = {'conv1_1':1,
				'conv2_1':0.75,
				'conv3_1':0.2,
				'conv4_1':0.2,
				'conv5_1':0.2,}
GM = utils.GramMatrix()
optimizer = optim.Adam([target],lr=0.01)

for epoch in range(1000):
	optimizer.zero_grad()
	content_features = vgg(content)
	style_features = vgg(style)
	target_features = vgg(target)

	style_gram_matrix = [GM(style_features[i]) for i in style_features if i !='conv4_2']
	target_gram_matrix = [GM(target_features[i]) for i in target_features if i !='conv4_2']

	style_loss = 0
	content_loss = 0.5*(torch.mean(content_features['conv4_2']-target_features['conv4_2']))**2
	for weight,i,j in zip(style_weights,target_gram_matrix,style_gram_matrix):
		style_loss += style_weights[weight]*torch.mean((i-j)**2)
	total_loss = alpha*content_loss+beta*style_loss
	total_loss.backward()
	optimizer.step()
	utils.save_img(target,args.output_file)
	print(f'Epoch {epoch+1} Loss {total_loss:.4f}')