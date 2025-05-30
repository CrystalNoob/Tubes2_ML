import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from keras.src.backend import Variable
import keras
import tensorflow as tf

class LSTMScratch():
    def __init__(
            self,
            units: int, # neuron
            keras_weight: list[Variable],
            activation: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.tanh, # activation function
            return_sequences = False
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
        self.h_t_total = []
        self.activation = activation
        self.return_sequences = return_sequences

    # def fit(self, input_feature: NDArray[np.float64]):
    #     self.input_feature = input_feature

    def forward(self, input_feature):
        if isinstance(input_feature, tf.Tensor):
            input_feature = input_feature.numpy() # type: ignore
        sample = input_feature.shape[0]
        timestep = input_feature.shape[1]
        for i in range(sample):
            current_sample = input_feature[i].copy()
            self.temp_h = [np.zeros(self.units)]
            self.temp_c = [np.zeros(self.units)]

            for j in range(timestep):
                h_t, c_t = self.calc(current_sample[j]) 
                self.temp_h.append(h_t)
                self.temp_c.append(c_t)

            self.h_t.append(self.temp_h[-1])
            self.h_t_total.append(self.temp_h)
            self.c_t.append(self.temp_c[-1])
        if self.return_sequences:
            h_sequences = [np.stack(h[1:], axis=0) for h in self.h_t_total]
            return np.stack(h_sequences, axis=0)
        else:
            return np.array(self.h_t) 

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