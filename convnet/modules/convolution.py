import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        N, C, H, W   = x.shape
        padding      = self.padding
        kernel_size  = self.kernel_size
        stride       = self.stride
        in_channels  = self.in_channels
        out_channels = self.out_channels
        weight = self.weight
        wn, _, wh, ww = weight.shape

        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        #output dimensions
        H_pad = H + 2 * padding
        W_pad = W + 2 * padding
        H_out = (H_pad - kernel_size) // stride + 1
        W_out = (W_pad - kernel_size) // stride + 1

        #strides
        stride_shape = (C, wh, ww, N, H_out, W_out)
        strides = (H_pad * W_pad, W_pad, 1, C * H_pad * W_pad, stride * W_pad, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_pad, shape=stride_shape, strides=strides)
        x_contig = np.ascontiguousarray(x_stride)
        x_contig.shape = (C * wh * ww, N * H_out * W_out)

        # Now all our convolutions are a big matrix multiply
        res = self.weight.reshape(wn, -1).dot(x_contig) + self.bias.reshape(-1, 1)

        # Reshape the output
        res.shape = (wn, N, H_out, W_out)
        out = res.transpose(1, 0, 2, 3)
        out = np.ascontiguousarray(out)

        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache

        weight=self.weight
        N, C, H, W = x.shape
        wn, _, wh, ww = weight.shape
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        in_channels = self.in_channels
        out_channels = self.out_channels
        _, _, H_out, W_out = dout.shape

        #db
        db = np.sum(dout, axis=(0, 2, 3))

        #dw
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        H_pad = H + 2 * padding
        W_pad = W + 2 * padding
        stride_shape = (C, wh, ww, N, H_out, W_out)
        strides = (H_pad * W_pad, W_pad, 1, C * H_pad * W_pad, stride * W_pad, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_pad, shape=stride_shape, strides=strides)
        x_contig = np.ascontiguousarray(x_stride)
        x_contig.shape = (C * wh * ww, N * H_out * W_out)
        dout_t = dout.transpose(1, 0, 2, 3).reshape(wn, -1)
        dw = dout_t.dot(x_contig.T).reshape(weight.shape)

        #dx
        dx_cols = weight.reshape(wn, -1).T.dot(dout_t)
        dx_cols.shape = (C, wh, ww, N, H_out, W_out)
        dx_padded = np.zeros((N, C, H_pad, W_pad),dtype=dx_cols.dtype)
        for n in range(N):
            for c in range(C):
                for kh in range(wh):
                    for kw in range(ww):
                        for h in range(H_out):
                            for w in range(W_out):
                                dx_padded[n, c, stride * h + kh, stride * w + kw] += dx_cols[c, kh, kw, n, h, w]
        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded
        self.dx=dx
        self.dw=dw
        self.db=db
        return dx, dw, db
