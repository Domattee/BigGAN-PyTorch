''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import numpy as np
import os

import torch
import torchvision

import utils
import BigGAN
import matplotlib.pyplot as plt

#	dir should point to the directory containing the G or G_ema weights.
def load(dir):
	strict = True
	params = {}
	params["G_ch"] = 96
	params["dim_z"] = 120
	params["hier"] = True
	params["shared_dim"] = 128
	params["G_shared"] = True
	params["skip_init"] = True
	params["no_optim"] = True
	params["G_activation"] = utils.activation_dict["inplace_relu"]
	
	G = BigGAN.Generator(**params).cuda()
	G.load_state_dict(torch.load(os.path.join(dir, "G_ema.pth")), strict=strict)
	#G.optim.load_state_dict(torch.load(os.path.join(dir, "G_optim.pth")), strict=strict)
	#state_dict = torch.load(os.path.join(dir, "state_dict.pth"))
	return G

	
	# Required a Generator object G. 
	#	im_class is an integer representing the imagenet class.
	#	n_images is the number of images to generate
	#	n_columns determines how many images will be saved in a single row
	#	filename is the path where the image will be stored, if None the image isn't saved.
def generate(G, im_class, n_images=1, filename=None, n_columns=2):
	device = "cuda"
	G.eval()
	
	y = torch.ones(n_images, dtype=torch.int64, device=device)*im_class
	z = torch.randn(n_images, G.dim_z, device=device)
	
	with torch.no_grad():
		out = G(z, G.shared(y))
	
	out = out.cpu()
	out = (out + 1) / 2.0 # out is between -1 and 1, we need to shift and scale to 0 to 1

	out = torchvision.utils.make_grid(out, nrow=n_columns)
	if not filename is None:
		torchvision.utils.save_image(out, filename)
	
	npout = out.numpy()
	
	plt.imshow(np.transpose(npout, (1,2,0)), interpolation='nearest')
	plt.show()
	
if __name__ == "__main__":
	G = load(os.path.join("..", "BigGAN_model"))
	generate(G, 933, n_images=10, n_columns=5)
	