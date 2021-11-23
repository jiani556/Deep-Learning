import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import GuidedGradCam, GuidedBackprop
from captum.attr import LayerActivation, LayerConductance, LayerGradCam

from data_utils import *
from image_utils import *
from captum_utils import *
import numpy as np

from visualizers import GradCam


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X, y, class_names = load_imagenet_val(num=5)

# FOR THIS SECTION ONLY, we need to use gradients. We introduce a new model we will use explicitly for GradCAM for this.
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
gc = GradCam()

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

# Guided Back-Propagation
gbp_result = gc.guided_backprop(X_tensor,y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gbp_result.shape[0]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(gbp_result[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_backprop.png')



# GradCam
# GradCAM. We have given you which module(=layer) that we need to capture gradients from, which you can see in conv_module variable below
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
for param in gc_model.parameters():
    param.requires_grad = True

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gradcam_val = gradcam_result[i]
    img = X[i] + (matplotlib.cm.jet(gradcam_val)[:,:,:3]*255)
    img = img / np.max(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/gradcam.png')


# As a final step, we can combine GradCam and Guided Backprop to get Guided GradCam.
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)
gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gbp_val = gbp_result[i]
    gbp_val /= np.max(gbp_val)
    gradcam_val = (matplotlib.cm.jet(gradcam_result[i])[:, :, :3] * 255)

    # Pointwise multiplication and normalization of the gradcam and guided backprop results (2 lines)
    img = gradcam_val * gbp_val
    img = img / np.max(img)

    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.float32(img)
    img = torch.from_numpy(img)
    img = deprocess(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_gradcam.png')


# **************************************************************************************** #
# Captum
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)

# Computing Guided GradCam
gc_result = GuidedGradCam(model,model.features[3])
attr_ggc = compute_attributions(gc_result,X_tensor,target = y_tensor)
attr_ggc = attr_ggc.abs()
attr_ggc,i= torch.max(attr_ggc, dim=1)
attr_ggc = attr_ggc.unsqueeze(0)
visualize_attr_maps('visualization/Captum_Guided_GradCam.png',X, y, class_names,attr_ggc, ['GuidedGradCam'], lambda attr:attr.detach().numpy())


# # Computing Guided BackProp
gbp_result = GuidedBackprop(model)
attr_gbp = compute_attributions(gbp_result,X_tensor,target = y_tensor)
attr_gbp = attr_gbp.abs()
attr_gbp,i= torch.max(attr_gbp, dim=1)
attr_gbp = attr_gbp.unsqueeze(0)
visualize_attr_maps('visualization/Captum_Guided_BackProp.png',X, y, class_names, attr_gbp, ['GuidedBackprop'],  lambda attr:attr.detach().numpy())


# Try out different layers
# and see observe how the attributions change
layer = model.features[3]

layer_act = LayerActivation(model, layer)
layer_act_attr = compute_attributions(layer_act, X_tensor)
layer_act_attr_sum = layer_act_attr.mean(axis=1, keepdim=True)

# Layer gradcam aggregates across all channels
layer_gradcam = LayerGradCam(model, layer)
layer_gradcam_attr= compute_attributions(layer_gradcam, X_tensor, target=y_tensor, relu_attributions=True)
layer_gradcam_attr_sum = layer_gradcam_attr.mean(axis = 1, keepdim = True)
layer_gradcam_attr_sum = layer_gradcam_attr_sum.permute(1,0,2,3)
visualize_attr_maps('visualization/layer_gradcam.png', X, y, class_names, layer_gradcam_attr_sum, ['layer_gradcam'],
                    lambda attr:attr.detach().numpy())

layer_conduct = LayerConductance(model,layer)
layer_conduct_attr = compute_attributions(layer_conduct, X_tensor, target = y_tensor)
layer_conduct_attr_sum = layer_conduct_attr.mean(axis = 1, keepdim = True)
layer_conduct_attr_sum = layer_conduct_attr_sum.permute(1,0,2,3)
visualize_attr_maps('visualization/layer_conduct.png', X, y, class_names, layer_conduct_attr_sum, ['layer_conductance'],
                    lambda attr:attr.detach().numpy())
