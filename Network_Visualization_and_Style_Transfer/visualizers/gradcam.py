import torch
from torch.autograd import Function as TorchFunc
import numpy as np
from PIL import Image


# The ’deconvolution’ is equivalent to a backward pass through the network, except that
# when propagating through a nonlinearity, its gradient is solely computed based on the
# top gradient signal, ignoring the bottom input. In case of the ReLU nonlinearity this
# amounts to setting to zero certain entries based on the top gradient. We propose to
# combine these two methods: rather than masking out values corresponding to negative
# entries of the top gradient (’deconvnet’) or bottom data (backpropagation), we mask
# out the values for which at least one of these values is negative.

class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, y):

        grad = None
        x, output = self.saved_tensors
        grad = torch.addcmul(torch.zeros(y.size()), y, (y > 0).type_as(y))
        grad[x <= 0] = 0

        return grad


class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        for param in gc_model.parameters():
            param.requires_grad = True
        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply

        scores = gc_model(X_tensor)
        score_class = scores.gather(1, y_tensor.unsqueeze(0).transpose(0, 1)).squeeze()
        score_class.sum().backward()
        grad = X_tensor.grad
        grad = grad.permute(0, 2, 3, 1).detach().numpy()
        return grad

    def grad_cam(self, X_tensor, y_tensor, gc_model):
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)
        scores = gc_model(X_tensor)
        score_class = scores.gather(1,y_tensor.unsqueeze(0).transpose(0,1)).squeeze()
        score_class.sum().backward()
        N, C, H, W = self.gradient_value.size()
        avg_gradient_value = self.gradient_value.view(N, C, -1).sum(2)
        avg_gradient_value = (avg_gradient_value / (H * W))[:, :, None, None]
        cam = (avg_gradient_value * self.activation_value).sum(1).detach().numpy()

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
