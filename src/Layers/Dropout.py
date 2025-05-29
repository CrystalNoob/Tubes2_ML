import numpy as np

class Dropout:
    def __init__(self, rate, seed=None):
        if not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`rate`. Expected a float value between 0 and 1. "
                f"Received: rate={rate}"
            )
        self.rate = rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def forward(self, x, training=False):
        if not training:
            return x

        keep_prob = 1.0 - self.rate
        self.mask = self.rng.uniform(size=x.shape) < keep_prob

        return (x * self.mask) / keep_prob
