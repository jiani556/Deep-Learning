# Network Visualization
- explore the use of different type of attribution algorithms - both gradient and perturbation - for images, and understand their differences using the Captum model interpretability tool for PyTorch

## You should reuse your conda environment in convent with the following additional
- packages installed.
- $ pip install future
- $ pip install scipy
- $ pip install torchvision

## Saliency Map
- A saliency map tells us the degree to which each pixel in the image affects the classification score for that image.

## GradCam
- GradCAM (which stands for Gradient Class Activation Mapping) is a technique that tells us where a convolutional network is looking when it is making a decision on a given input image.

## Fooling Image
-  We can also use the similar concept of image gradients to study the stability of the network. Consider a state-of-the-art deep neural network that generalizes well on an object recognition task. We expect such network to be robust to small perturbations of its input, because small perturbation cannot change the object category of an image.
- Given an image and a target class, we can perform gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class. We term the so perturbed examples “adversarial examples”.

##Class Visualization
- By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class.

# Style Transfer
- The general idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic ”style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep network, and then performing gradient descent on the pixels of the image itself.

## Content Loss
- We can generate an image that reflects the content of one image and the style of another by incorporating both in our loss function.

## Style Loss
- We can tackle the style loss

## Total Variation Loss
It turns out that it’s helpful to also encourage smoothness in the image. We can do this by adding another term to our loss that penalizes wiggles or **total variation** in the pixel values. This concept is widely used in many computer vision task as a regularization term.
