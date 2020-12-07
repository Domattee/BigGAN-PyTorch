import numpy as np
import os

import torch
import torchvision

import utils
import BigGAN
import matplotlib.pyplot as plt


params = {}
# Cuda
params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# Number of classes in model. 1000 for imagenet
params["classes"] = 1000
# Architecture params
params["G_ch"] = 96
params["dim_z"] = 120
params["hier"] = True
params["shared_dim"] = 128
params["G_shared"] = True
params["skip_init"] = True
params["no_optim"] = True
params["G_activation"] = utils.activation_dict["inplace_relu"]
	

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

	# Use GPU if possible
	if params["device"] == "cuda":
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
	show:		Boolean, default True. Displays the image with pyplot if true.
	
Returns:
	The generators output scaled to between 0 and 1 if show = False, else None.
	
"""
#TODO: truncate, commandline parser
def generate(G, im_class, n_images=1, n_columns=2, filename=None, show = True):
	# Put into eval mode
	G.eval()
	
	# y is the class vector, z the random noise vectors.
	y = torch.ones(n_images, dtype=torch.int64, device=params["device"])*im_class
	z = torch.randn(n_images, G.dim_z, device=params["device"]) # z is sampled from N(0,1)
	
	# Generate images
	with torch.no_grad():
		out = G(z, G.shared(y))
	
	return show_or_save(out, filename, show, n_columns)

	
### Taken from utils, but altered to allow variable device. 
### Interpolates tensors with format BIX where I is the dim along which interps are stored.
def interps(x0, x1, n_interps):
	lerp = torch.linspace(0, 1.0, n_interps, device=params["device"]).to(x0.dtype)
	return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

	
"""
Generates images which are interpolations between two classes A and B. 
Interpolations are saved as a grid with each set as one row

Args:
	G:			Generator Object
	class_A:	First image class as an Integer.
	class_B:	Second image class as an Integer.
	n_interps:	How many interpolation points. Start and end points are included. Integer >= 2.
	n_samples:	How many different sets of start and end points we generate. 
				Also determines the number of rows in the resulting image.
	filename:	If provided, saves the image grid to this file.
	show:		Boolean, default True. Displays the image with pyplot if true.
	fix_z:		Boolean, default True. If True only one noise vector is generated for each set.
	
Returns:
	The generators output scaled to between 0 and 1 if show = False, else None.
	
"""
def interpolate(G, class_A, class_B, n_interps=5, n_samples=2, filename=None, show=True, fix_z = True):
	# Put into eval mode
	G.eval()
	
	if fix_z:
		z = torch.randn(n_samples, 1, G.dim_z, device=params["device"])
		zs = z.repeat(1, n_interps, 1)
	else:
		z_A = torch.randn(n_samples, 1, G.dim_z, device=params["device"])
		z_B = torch.randn(n_samples, 1, G.dim_z, device=params["device"])
		zs = interps(z_A, z_B, n_interps)
	z = torch.flatten(zs, start_dim=0, end_dim=1)
	
	y_A = torch.ones(n_samples, dtype=torch.int64, device=params["device"])*class_A
	y_B = torch.ones(n_samples, dtype=torch.int64, device=params["device"])*class_B
	ys = interps(G.shared(y_A).unsqueeze(1), G.shared(y_B).unsqueeze(1), n_interps)
	y = torch.flatten(ys, start_dim=0, end_dim=1)
	
	# Generate images
	with torch.no_grad():
		out = G(z, y)
		
	return show_or_save(out, filename, show, n_interps)
		
		
"""
Utility function which takes a generator output, scales it into proper image range, turns it into a grid and then saves or displays that grid.

Args:
	out:		pytorch tensor containing output from the generator in a BCHW format.
	filename:	If provided, saves the image grid to this file.
	show:		Boolean, default True. Displays the image with pyplot if true.
	n_columns:	The number of columns in the final image grid.
	
Returns:
	The generators output scaled to between 0 and 1 if show = False, else None.
"""
def show_or_save(out, filename=None, show=True, n_columns = 5):
	if params["device"] == "cuda":
		out = out.cpu()
		
	out = (out + 1) / 2.0 # out is between -1 and 1, we need to shift and scale to 0 to 1
	
	# Make images into grid
	out = torchvision.utils.make_grid(out, nrow=n_columns, padding=10)
	
	# Optionally save image
	if not filename is None:
		torchvision.utils.save_image(out, filename)
	
	if not show:
		return out

	# Make numpy for display with pyplot
	npout = out.numpy()
	
	# Display image grid. Grid output is (channels, width, height), but pyplot wants width, height, channels so we have to shuffle dimensions
	plt.figure()
	plt.imshow(np.transpose(npout, (1,2,0)), interpolation='nearest')
	plt.show()
	
	
# CHEESEBURGERS
if __name__ == "__main__":
	G = load(os.path.join("..", "BigGAN_model"), "G_ema.pth")
	plt.ion()
	generate(G, 933, n_images=10, n_columns=5)
	interpolate(G, 2, 933, n_interps=5, n_samples=3)
	plt.show()
	input("Press enter to continue...")
