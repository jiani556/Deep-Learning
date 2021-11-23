import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        N, C, H, W = features.size()
        # FL
        F_l = features.view(1, C, H * W)
        # Gram
        gram = torch.mm(F_l[0, :, :], F_l[0, :, :].transpose(1, 0))
        if normalize:
            gram = gram/ (H * W * C)
        # # Add back first dimension
        gram = gram.unsqueeze(0)
        return gram

    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of the same length as style_layers, where style_targets[i] is
             a PyTorch Variable giving the Gram matrix the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """
        style_loss = torch.zeros(1)
        for i in range(len(style_layers)):
            im_gram = self.gram_matrix(feats[style_layers[i]],normalize=True)
            style_loss += style_weights[i] * torch.sum((im_gram - style_targets[i]) ** 2)
        return style_loss
