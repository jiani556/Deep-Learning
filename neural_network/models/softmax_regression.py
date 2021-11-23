import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        W1, G1 = self.weights['W1'], self.gradients['W1']
        Z1 = X.dot(W1)
        A1 = _baseNetwork.ReLU(self,Z1)
        score = _baseNetwork.softmax(self,A1)
        loss = _baseNetwork.cross_entropy_loss(self,score,y)
        accuracy = _baseNetwork.compute_accuracy(self,score,y)

        if mode != 'train':
            return loss, accuracy

        dscores = score
        dscores[np.arange(X.shape[0]), y] -= 1
        dscores = dscores/X.shape[0]
        dscores[Z1<=0] =0
        self.gradients['W1'] = X.T.dot(dscores)

        return loss, accuracy
