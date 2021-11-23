import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, Saliency

from visualizers import SaliencyMap
from data_utils import *
from image_utils import *
from captum_utils import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

X, y, class_names = load_imagenet_val(num=5)
# manually compute saliency maps
sm = SaliencyMap()
sm.show_saliency_maps(X, y, class_names, model)

# ************************************************************************************** #

# use Captum for saliency map

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)

# Computing Integrated Gradient
int_grads = IntegratedGradients(model)
attr_ig = compute_attributions(int_grads, X_tensor, target=y_tensor, n_steps=10)
##############################################################################
# TODO: Compute/Visualize Saliency using captum.                             #
#       visualize_attr_maps function from captum_utils.py is useful for      #
#       visualizing captum outputs                                           #
##############################################################################
# Computing saliency maps
#convert to 1d take max

saliency = attr_ig.abs()
# saliency = attr_ig
saliency,i= torch.max(saliency, dim=1)
saliency = saliency.unsqueeze(0)
visualize_attr_maps('visualization/captum_saliency_map.png', X, y, class_names, saliency, ['saliency'], lambda attr: attr.numpy())

# saliency = attr_ig.permute(1,0,2,3)
# visualize_attr_maps('./', X, y, class_names, saliency, ['c1','c2','c3'], lambda attr: attr.numpy())
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
