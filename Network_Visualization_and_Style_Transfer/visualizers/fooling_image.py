import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize our fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        # We will fix these parameters for everyone so that there will be
        # comparable outputs

        learning_rate = 10  # learning rate is 1
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):

            # Forward
            scores = model(X_fooling_var)
            # check if fool the model
            _, max_index = scores.data.max(dim=1)
            if max_index[0] == target_y:
                break
            # Scores of target_y
            target_score = scores[0, target_y]
            # Backward
            target_score.backward()
            # Gradient
            im_grad = X_fooling_var.grad.data
            # update x with normalised gradient
            X_fooling_var.data += learning_rate * (im_grad / im_grad.norm())
            X_fooling_var.grad.data.zero_()
            
        X_fooling = X_fooling_var.data

        return X_fooling
