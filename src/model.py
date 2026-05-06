# src/model.py

import torch.nn as nn

class ChargePredictorNet(nn.Module):
    """
    A fully connected neural network for predicting the magnitudes of two source charges
    given positional and field/force information. The architecture and activation
    functions are configurable via constructor arguments to enable experimentation
    with deeper networks, alternative nonlinearities and output range constraints.

    Args:
        input_dim (int): Number of input features. Defaults to 15 for the three-charge
            dataset (positions of charges, total electric field and force components).
        hidden_dims (list[int]): Sizes of the hidden layers. Increasing the depth
            or width of the network can significantly improve modelling capacity.
        output_dim (int): Size of the output layer. For two unknown charges this
            should be 2.
        dropout_rate (float): Dropout probability applied after each activation.
            Lower dropout (e.g. 0.1) can improve performance on synthetic data where
            overfitting is less of a concern. Defaults to 0.2.
        use_batchnorm (bool): If True, includes a BatchNorm1d layer after each
            linear layer. Batch normalisation can accelerate convergence and
            stabilise training when working with features of varying scales.
        activation (str): Name of the activation function to use in the hidden
            layers. Supported options are 'relu', 'leaky_relu' and 'elu'. Defaults
            to 'relu'.
        output_activation (bool): If True, applies a tanh activation to the
            network output and rescales it to the specified `q_range`. This
            constrains the predicted charges to lie within ±q_range, which makes
            sense when training on data drawn from a bounded uniform distribution.
        q_range (float): Maximum absolute value of the charge in the dataset.
            When `output_activation` is True, the final outputs will be in
            [-q_range, q_range]. Defaults to 5.0 for the provided dataset.

    Example:

        model = ChargePredictorNet(
            input_dim=15,
            hidden_dims=[512, 256, 128],
            activation='elu',
            dropout_rate=0.1,
            use_batchnorm=True,
            output_activation=True,
            q_range=5.0
        )
    """

    def __init__(self,
                 input_dim: int = 15,
                 hidden_dims: list = None,
                 output_dim: int = 2,
                 dropout_rate: float = 0.2,
                 use_batchnorm: bool = False,
                 activation: str = 'relu',
                 output_activation: bool = False,
                 q_range: float = 5.0):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        # Map string names to PyTorch activation modules. Extendable for future
        # experimentation. Defaults to ReLU if an unknown name is supplied.
        activation = (activation or 'relu').lower()
        act_map = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
            'elu': nn.ELU
        }
        act_cls = act_map.get(activation, nn.ReLU)
        act_fn = act_cls()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            # Linear transformation
            layers.append(nn.Linear(in_dim, h))
            # Normalisation layer, if requested
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            # Nonlinearity
            layers.append(act_fn)
            # Dropout to mitigate overfitting
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            # update dimension for next layer
            in_dim = h
        # Final linear layer to produce raw outputs
        layers.append(nn.Linear(in_dim, output_dim))
        # Optional output activation: tanh ensures outputs are in [-1,1], then
        # scaled to the dataset charge range.
        self._apply_output_activation = bool(output_activation)
        self.q_range = float(q_range)
        if self._apply_output_activation:
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        # If output activation is enabled, rescale the tanh outputs to
        # the configured charge range. Otherwise, return raw predictions.
        if self._apply_output_activation:
            return out * self.q_range
        return out
