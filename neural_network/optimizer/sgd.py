from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :param gradient: The Gradient computed in forward step
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)
        for x in model.weights:
            model.weights[x] -= self.learning_rate * model.gradients[x]
