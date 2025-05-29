import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from keras.src.backend import Variable
import keras
import tensorflow as tf

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
            keras_weight: list[Variable],
            activation: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.tanh, # activation function
    ):
        # kernel = W, recurrent_kernel = U, bias = b
        # kernel : (input_dim × 4 × units) 
        # recurrent kernel : (units × 4 × units)
        # bias : (4 × units)
        # f = sigmoid(Wf * input + Uf * ht-1 + bf)
        # i = sigmoid(Wi * input + Ui * ht-1 + bi)
        # ~c = activation_func(Wc * input + Uc * ht-1 + bc)
        # c = f * ct-1 + i * ~c
        # o = sigmoid(Wo * input + Uo * ht-1 + bo)
        # h = ot * activation_func(c)
        self.units = units
        self.kernel, self.recurrent_kernel, self.bias = keras_weight
        self.h_t = []
        self.c_t = []
        self.activation = activation

    def fit(self, input_feature: NDArray[np.float64]):
        self.input_feature = input_feature

    def predict(self, input_feature):
        if isinstance(input_feature, tf.Tensor):
            input_feature = input_feature.numpy() # type: ignore
        sample = input_feature.shape[0]
        timestep = input_feature.shape[1]
        for i in range(sample):
            current_sample = input_feature[i]
            self.temp_h = [np.zeros(self.units)]
            self.temp_c = [np.zeros(self.units)]

            for j in range(timestep):
                h_t, c_t = self.calc(current_sample[j]) 
                self.temp_h.append(h_t)
                self.temp_c.append(c_t)

            self.h_t.append(self.temp_h[-1])
            self.c_t.append(self.temp_c[-1])
        return self.h_t

    def calc(self, x_t: NDArray[np.float64]):
        w_dot_x = x_t @ self.kernel
        u_dot_h = self.temp_h[-1] @ self.recurrent_kernel
        result = w_dot_x + u_dot_h + self.bias 
        k_i, k_f, k_c, k_o = np.split(result, 4, axis=0)

        i_t, f_t, o_t = self.sigmoid(k_i), self.sigmoid(k_f), self.sigmoid(k_o)

        c_tilde = self.activation(k_c)
        c_t = f_t * self.temp_c[-1] + i_t * c_tilde
        
        h_t = o_t * self.activation(c_t)

        return h_t, c_t     

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


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