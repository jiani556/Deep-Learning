import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

from image_utils import preprocess
class SaliencyMap:
    def compute_saliency_maps(self, X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Wrap the input tensors in Variables
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        saliency = None

        lam = 1e3  # This is the regularization parameter when you need it

        #forward
        scores = model(X_var)
        scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
        #backward
        scores.backward(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))
        #gradient
        gradient = X_var.grad.data
        #convert to 1d take max
        saliency = gradient.abs()
        saliency, i = torch.max(saliency, dim=1)
        saliency = saliency.squeeze()
        return saliency

    def show_saliency_maps(self, X, y, class_names, model):
        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self.compute_saliency_maps(X_tensor, y_tensor, model)
        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()

        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(X[i])
            plt.axis('off')
            plt.title(class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.gray)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.savefig('visualization/saliency_map.png')
        plt.show()
