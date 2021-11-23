import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        #    1)forward process:                                                     #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        W1, b1 = self.weights['W1'], self.weights['b1']
        W2, b2 = self.weights['W2'], self.weights['b2']

        h1 = X.dot(W1) + b1
        a1 = _baseNetwork.sigmoid(self, h1)
        h2 = a1.dot(W2) + b2
        score = _baseNetwork.softmax(self, h2)
        loss = _baseNetwork.cross_entropy_loss(self, score, y)
        accuracy = _baseNetwork.compute_accuracy(self, score, y)

        #############################################################################
        #    backward process:                                                   ``#
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        dscores = score
        dscores[np.arange(X.shape[0]), y] -= 1
        dscores=dscores/X.shape[0]
        # W2 and b2
        self.gradients['W2'] = np.dot(a1.T, dscores)
        self.gradients['b2'] = np.sum(dscores, axis=0)
        # backprop into hidden layer
        hlayer = np.dot(dscores, W2.T)
        # backprop the sigmoid non-linearity
        hlayer = hlayer*_baseNetwork.sigmoid_dev(self,h1)
        # finally into W,b
        self.gradients['W1'] = X.T.dot(hlayer)
        self.gradients['b1'] = np.sum(hlayer, axis=0)
        return loss, accuracy
