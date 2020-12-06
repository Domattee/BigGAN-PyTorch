import numpy as np
import os

import torch
import torchvision

import utils
import BigGAN
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Loads a pretrained model from dir.

Args:
	dir:	String containing the path to the directory with the .pth files
	model:	The specific .pth file that should be loaded. Must match the BigGan generator.
	
Returns:
	BigGan Generator object

"""
def load(dir, model):
	# Parameters which were used for the pretrained model
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
	
	# Use GPU if possible
	if device == "cuda":
		G = BigGAN.Generator(**params).cuda()
	else:
		G = BigGAN.Generator(**params).cpu()
		
	# Load the weights
	G.load_state_dict(torch.load(os.path.join(dir, model)), strict=strict)
	return G

	
"""
Generates a number of images of a given class and displays them in a grid. Optionally also saves them to file.

Args:
	G:			Generator Object
	im_class:	Image class as an integer.
	n_images:	The number of images to generate. Integer.
	n_columns:	How many columns should be in the final display grid.
	filename:	If provided, saves the image grid to this file.
	
Returns:

"""
def generate(G, im_class, n_images=1, filename=None, n_columns=2):
	# Put into eval mode
	G.eval()
	
	# y is the class vector, z the random noise vectors.
	y = torch.ones(n_images, dtype=torch.int64, device=device)*im_class
	z = torch.randn(n_images, G.dim_z, device=device) # z is sampled from N(0,1)
	
	# Generate images
	with torch.no_grad():
		out = G(z, G.shared(y))
	
	
	if device == "cuda":
		out = out.cpu()
		
	out = (out + 1) / 2.0 # out is between -1 and 1, we need to shift and scale to 0 to 1
	
	# Make images into grid
	out = torchvision.utils.make_grid(out, nrow=n_columns)
	
	# Optionally save image
	if not filename is None:
		torchvision.utils.save_image(out, filename)
	
	# Make numpy for display with pyplot
	npout = out.numpy()
	
	# Display image grid. Grid output is (channels, width, height), but pyplot wants width, height, channels so we have to shuffle dimensions
	plt.imshow(np.transpose(npout, (1,2,0)), interpolation='nearest')
	plt.show()
	
# CHEESEBURGERS
if __name__ == "__main__":
	G = load(os.path.join("..", "BigGAN_model"), "G_ema.pth")
	generate(G, 933, n_images=10, n_columns=5)
	