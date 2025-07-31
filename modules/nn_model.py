""" Module containing functions to define standard and Bayesian neural networks 

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v2 as tfc
import tensorflow_probability as tfp

tfc.enable_v2_behavior()
tfd = tfp.distributions

##  Standard neural network with TensorFlow
def MLP_TF(p, dropout_rate, hidden_layers=1, nodes_per_layer=None, optimizer='adam'):
    """ Returns a Tensorflow neural network object. The default neural network consists of a normalized input layer
    of size p, a single normalized hidden layer of size p with dropout and ReLU activation, and an output layer of
    size 1. Number and size of hidden layers can be changed.

    Parameters
    ----------
    p : int, mandatory
        The size of the input vector

    dropout_rate: float between [0, 1], mandatory
        The dropout rate in the hidden layer

    hidden_layers : int, optional (default=1)
        Number of hidden layers.

    nodes_per_layer : int or list of ints, optional (default=None)
        Number of nodes per hidden layer.
        - If int, all hidden layers will have the same number of nodes.
        - If list, must have length equal to hidden_layers.


    """

    if nodes_per_layer is None:
        # Default: hidden layers have p nodes each
        nodes_per_layer = [p] * hidden_layers
    elif isinstance(nodes_per_layer, int):
        nodes_per_layer = [nodes_per_layer] * hidden_layers
    elif isinstance(nodes_per_layer, list):
        if len(nodes_per_layer) != hidden_layers:
            raise ValueError("Length of nodes_per_layer list must equal hidden_layers")

    model = keras.models.Sequential()
    model.add(keras.Input(shape=(p,)))
    model.add(keras.layers.LayerNormalization())

    for nodes in nodes_per_layer:
        model.add(keras.layers.Dense(nodes))
        model.add(keras.layers.LayerNormalization())
        model.add(keras.layers.ReLU())
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1))  # output layer
    model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])

    return model


##  Bayesian neural network with Tensorflow Probability
def negloglik(y, yhat):
    """ Loss function for the negative log likelhood loss used to train a Bayesian neural network
    with TensorFlow Probability

    Parameters
    ----------
    y : tensor, mandatory
        The true value of y
    yhat: tensor, mandatory
        The predicted value of y

    """

    return -yhat.log_prob(y)


def MLP_TFP(p, kl_loss_weight, feat):
    """ Returns a TensorFlow Probability neural network object. 
    The nested functions allows to define prior and posterior distributions
    with feature-dependent prior scales.
   
    Parameters
    ----------
    p : int, mandatory
        The size of the input vector
    kl_loss_weight: float, mandatory
        The weight for the Kullback-Leibler divergence loss
    feat: str ['rdkit', 'maccs', 'cddd', 'morgan-512'], mandatory
        The molecular descriptor type used to set the scale of the non-trainable prior.

    """

    # - specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1e-5))

        return tfc.keras.Sequential([tfp.layers.VariableLayer(2 * n, dtype=dtype),
                                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                        tfd.Normal(loc=t[..., :n],
                                                   scale=1e-5 + tfc.nn.softplus(c + t[..., n:])),
                                        reinterpreted_batch_ndims=1))])

    # - specify a non-trainable prior
    prior_scale = {'rdkit': 0.5,
                   'cddd': 0.5,
                   'morgan-512': 1,
                   'maccs': 1}

    def prior_non_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        mix = 0.5
        return tfc.keras.Sequential([tfp.layers.DistributionLambda(lambda t: tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1 - mix]),
            components=[tfd.MultivariateNormalDiag(loc=tfc.zeros(n), scale_diag=1e-6 * tfc.ones(n)),
                        tfd.MultivariateNormalDiag(loc=tfc.zeros(n), scale_diag=prior_scale[feat] * tfc.ones(n))])
                                                                  )])
    # - define model
    model = tfc.keras.models.Sequential()
    model.add(tfc.keras.Input(shape=(p,)))
    model.add(tfc.keras.layers.LayerNormalization())
    model.add(tfp.layers.DenseVariational(p, posterior_mean_field, prior_non_trainable, kl_weight=kl_loss_weight,
                                          activation='relu'))
    model.add(tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_non_trainable, kl_weight=kl_loss_weight))
    model.add(tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1], scale=1e-5 + tfc.math.softplus(0.1 * t[..., 1:]))))

    return model
