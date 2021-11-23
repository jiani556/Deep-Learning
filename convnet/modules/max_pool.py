import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        x_reshaped = x.reshape(N, C, H // kernel_size, kernel_size, W // kernel_size, kernel_size)
        out = x_reshaped.max(axis=3).max(axis=4)
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        #output from forward
        x_reshaped = x.reshape(N, C, H // kernel_size, kernel_size, W // kernel_size, kernel_size)
        out = x_reshaped.max(axis=3).max(axis=4)
        #dx after max_pooling
        dx_reshaped = np.zeros_like(x_reshaped)
        out_reshaped = out[:, :, :, np.newaxis, :, np.newaxis]
        match = (x_reshaped == out_reshaped)
        dout_reshaped  = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_bc, _ = np.broadcast_arrays(dout_reshaped, dx_reshaped)
        dx_reshaped[match] = dout_bc[match]
        dx_reshaped /= np.sum(match, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)
        self.dx=dx
        return dx
    
