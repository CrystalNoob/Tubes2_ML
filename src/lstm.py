import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from keras.src.backend import Variable
import keras

class LSTMScratch():
    '''
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    use_cudnn="auto",
    **kwargs
    '''
    def __init__(
            self,
            units: int, # neuron
            # input_feature: NDArray[np.float64], # input
            keras_weight: list[Variable],
            activation: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.tanh, # activation function
    ):
        # kernel = W, recurrent_kernel = U, bias = b
        # kernel : (input_dim × 4 × units) 
        # recurrent kernel : (units × 4 × units)
        # bias : (4 × units)
        self.kernel, self.recurrent_kernel, self.bias = keras_weight
        # f = sigmoid(Wf * input + Uf * ht-1 + bf)
        # i = sigmoid(Wi * input + Ui * ht-1 + bi)
        # ~c = activation_func(Wc * input + Uc * ht-1 + bc)
        # c = f * ct-1 + i * ~c
        # o = sigmoid(Wo * input + Uo * ht-1 + bo)
        # h = ot * activation_func(c)
        print(1)

    def forward(self):
        timestep = 1

        pass

    def sigmoid(self, x):
        return x


'''
    def calc(self, x_t: NDArray[np.float64]):
        w_dot_x = x_t @ self.kernel
        w_dot_x += self.bias
        print(w_dot_x)
        k_i, k_f, k_c, k_o = np.split(w_dot_x, 4, axis=0)
        w_i, w_f, w_c, w_o = np.split(self.kernel, 4, axis=1)
        b_i, b_f, b_c, b_o = np.split(self.bias, 4)
        p_i = x_t @ w_i
        p_f = x_t @ w_f
        p_c = x_t @ w_c
        p_o = x_t @ w_o
        print("=============")
        print(k_i)
        print(k_f)
        print(k_c)
        print(k_o)
        print("ini p")
        print(p_i+b_i)
        print(p_f+b_f)
        print(p_c+b_c)
        print(p_o+b_o)
        print()
        pass

'''