import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

class RNNScratch:
    def __init__(self,
                 input_feature: NDArray[np.float64],
                 input_weight: NDArray[np.float64],
                 hidden_weight: NDArray[np.float64],
                 hidden_bias_weight: NDArray[np.float64],
                 activation: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.tanh
                ) -> None:

        # Input stuff
        self.input_feature = input_feature
        self.input_weight = input_weight

        # Hidden stuff
        self.hidden_weight = hidden_weight
        self.bias_weight = hidden_bias_weight

        # Activation function with tanh as the default activation func
        self.activation = activation

    def forward(self, h_state_prev: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """
        I was dumb, just return h after n timestep, no need output
        ### h_t = f(U . x_t + W . h_t-1 + b_xh)

        Need to write this for my forgetful brain:
        U : Input weight matrix
        W : Hidden weight matrix
        x_t : Feature vector (i features) at step t
        h_t : Hidden state
        f: activation function
        """

        # input_features = total timestep
        total_timestep: int = self.input_feature.shape[0]
        h_state_prev = h_state_prev if h_state_prev is not None else np.zeros(self.bias_weight.shape[0])

        h = np.zeros((total_timestep, self.bias_weight.shape[0]))

        for t in range(total_timestep):
            h_t = self.activation(
                np.matmul(self.input_weight, self.input_feature[t]) +
                np.matmul(self.hidden_weight, h_state_prev) +
                self.bias_weight
            )

            h[t] = h_t
            h_state_prev = h_t

        return h
