import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# data = [
#     [10, 20, 30, 40, 50],
#     [60, 70, 80, 90, 100],
#     [110, 120, 130, 140, 150],
#     [160, 170, 180, 190, 200],
#     [210, 220, 230, 240, 250],
# ]

# data = np.array(data)

# kernel = [
#     [1, 0, -1],
#     [1, 0, -1],
#     [1, 0, -1],
# ]

# kernel = np.array(kernel)
# result = sliding_window_view(data,window_shape=(3,3))
# print(result[0])

# rez = np.array([[x.sum() for x in y] for y in result*kernel])

class Conv2DLayer:
    def __init__(self, kernel, bias=None, padding='same', activation=None):
        self.kernel = kernel
        self.bias = bias if bias is not None else np.zeros(kernel.shape[-1])
        self.padding = padding
        self.activation = activation
        
    def _apply_activation(self, x):
        if self.activation is None:
            return x
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'softmax':
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
    def forward(self, x):
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        _, _, _, C = x.shape
        kh, kw, C_in, _ = self.kernel.shape
        assert C == C_in, "Input channels must match kernel"
        pad_h, pad_w = (kh // 2, kw // 2) if self.padding == 'same' else (0, 0)
        x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        window = sliding_window_view(x_padded, window_shape=(kh, kw), axis=(1, 2))
        # bhwclk -> bhwklc
        window = window.transpose(0,1,2,4,5,3)
        out = np.einsum('bhwklc,klco->bhwo', window, self.kernel)
        # Bingung bang? LMAO
        # setiap gambar (1 input dari batch), setiap posisi x dan y,
        # dan setiap output channel punya output sendiri (Untuk setiap B H W O)
        # kernel height kernel width di k l dan input channel di c
        # Here lemme show u
        # for b in range(batch count):
        #     for h in range(input height):
        #         for w in range(input width):
        #             for o in range(output channels):
        #                 acc = 0
        #                 for k in range(kernel height):
        #                     for l in range(kernel width):
        #                         for c in range(input channels):
        #                             acc += window[b][h][w][k][l][c]*self.kernel[k][l][c][o]
        #                 out[b][h][w][o] = acc
        out += self.bias.reshape((1, 1, 1, -1))
        out = self._apply_activation(out)
        return out[0] if out.shape[0] == 1 else out



class Pooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='valid', type="max"):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.type = type

    def _pad_input(self, x):
        is_batch = x.ndim == 4
        if not is_batch:
            x = x[np.newaxis, ...]
        _, H, W, _ = x.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        if self.padding == 'same':
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))
            pad_h = max((out_h - 1) * sh + ph - H, 0)
            pad_w = max((out_w - 1) * sw + pw - W, 0)
            pad_top = pad_h // 2
            # Extra pixel
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            # Extra pixel
            pad_right = pad_w - pad_left
            # Biar ga terpengaruh kalau ada yg negatif utk max
            pad_val = -np.inf if self.type == "max" else 0.0
            x = np.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=pad_val
            )
        elif self.padding != 'valid':
            raise ValueError(f"Unsupported padding: {self.padding}")
        return x

    def forward(self, x):
        is_batch = x.ndim == 4
        if not is_batch:
            x = x[np.newaxis, ...]
        x = self._pad_input(x)
        ph, pw = self.pool_size
        sh, sw = self.strides
        windows = sliding_window_view(x, window_shape=(ph, pw), axis=(1, 2))
        # ambil setiap sh-kali dan setiap sw-kali
        windows = windows[:, ::sh, ::sw, :, :, :]
        # bhwclk -> bhwklc
        windows = windows.transpose(0, 1, 2, 4, 5, 3)
        if self.type == "max":
            out = np.max(windows, axis=(3, 4))
        elif self.type == "avg":
            out = np.mean(windows, axis=(3, 4))
        else:
            raise ValueError("Unknown pooling type")
        return out[0] if not is_batch else out

class FlattenLayer:
    def forward(self, x):
        if x.ndim == 4:
            return x.reshape(x.shape[0], -1)
        elif x.ndim == 3:
            return x.reshape(1, -1)
        else:
            raise ValueError(f"FlattenLayer expects 3D or 4D input, got shape {x.shape}")

        
class DenseLayer:
    def __init__(self, W, b, activation=None):
        self.W = W
        self.b = b
        self.activation = activation

    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        z = x @ self.W + self.b
        if self.activation is None:
            return z
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return e_z / e_z.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")