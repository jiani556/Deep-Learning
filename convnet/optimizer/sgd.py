from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        #self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                if hasattr(m, 'vw'):
                    v = m.vw
                    m.vw = self.momentum * v - self.learning_rate * m.dw
                else:
                    m.vw = - self.learning_rate * m.dw  # integrate velocity
                m.weight += m.vw

            if hasattr(m, 'bias'):
                if hasattr(m, 'vb'):
                    m.vb = self.momentum * m.vb - self.learning_rate * m.db
                else:
                    m.vb = -self.learning_rate * m.db
                m.bias += m.vb
