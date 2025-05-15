import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

class RNNScratch:
    def __init__(self,
                 input_feature: NDArray[np.float64],
                 input_weight: NDArray[np.float64],
                 hidden_weight: NDArray[np.float64],
                 hidden_bias_weight: NDArray[np.float64],
                 output_weight: NDArray[np.float64],
                 output_bias_weight: NDArray[np.float64],
                 activation: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.tanh
                ) -> None:

        # Input stuff
        self.input_feature = input_feature
        self.input_weight = input_weight

        # Hidden stuff
        self.hidden_weight = hidden_weight
        self.hidden_bias_weight = hidden_bias_weight
        self.hidden_dim = self.hidden_weight.shape[0]
        self.hidden_state = np.zeros(self.hidden_dim)

        # Output stuff
        self.output_weight = output_weight
        self.output_bias_weight = output_bias_weight

        # Activation function with tanh as the default activation func
        self.activation = activation

    def clear_hidden_state(self) -> None:
        self.hidden_state.fill(0)

    def forward(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        An idempotent function that does a forward pass once
        ### h_t = f(U . x_t + (W . h_t-1 + b_xh))
        ### y_t = f(V . h_t + b_hy)

        - x_t : feature vector (i features) at step t
        - h_t : hidden state.
        - y_t : output at step t.
        - f: activation function
        """

        h_t = self.activation(np.matmul(self.input_feature, self.input_weight) + (np.matmul(self.hidden_state, self.hidden_weight) + self.hidden_bias_weight))
        y_t = self.activation(np.matmul(h_t, self.output_weight) + self.output_bias_weight)

        return h_t, y_t
