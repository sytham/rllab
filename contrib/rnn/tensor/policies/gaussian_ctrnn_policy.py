import numpy as np

import lasagne.nonlinearities as NL
from contrib.rnn.tensor.policies.base import GaussianRNNPolicy
from contrib.rnn.tensor.network import EulerCTRNN
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import unflatten_tensors

class GaussianCTRNNPolicy(GaussianRNNPolicy):
    def create_mean_network(self, input_shape, output_dim, hidden_dim,
                            hidden_nonlinearity=NL.tanh,
                            output_nonlinearity=None,
                            dt=0.01, **kwargs):

        self.dt = dt
        return EulerCTRNN(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            **kwargs
        )
    
    @overrides
    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param.name == "tc":
                value = np.maximum(self.dt, value)
            param.set_value(value.astype(dtype))
            if debug:
                print("setting value of %s" % param.name)

