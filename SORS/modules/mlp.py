import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, TimeDistributed

@gin.configurable(module=__name__)
class MLP(Layer):
    def __init__(self,num_layers,dim,out_dim,activation='relu',name=None,in_dim=None,time_distributed=False,last_activation=None):
        """
        time_distributed:
            to support ragged Tensor; currently only support ragged tensor that is ragged for the 1-axis (second dimension)
        """
        super().__init__()

        self.layers = []
        for l in range(num_layers):
            l = Dense(
                dim,
                activation = activation,
                name = None if name is None else f'{name}_{l}',
            )

            if in_dim is not None:
                l.build((in_dim,))
                in_dim = dim

            if time_distributed:
                l = TimeDistributed(l)

            self.layers.append(l)

        l = Dense(
            out_dim,
            activation = last_activation,
            name = None if name is None else f'{name}_{num_layers}',
        )

        if in_dim is not None:
            l.build((in_dim,))

        if time_distributed:
            l = TimeDistributed(l)

        self.layers.append(l)

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o

    def decay_vars(self):
        # return only kernels without bias in the network
        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in self.layers
            ]

@gin.configurable(module=__name__)
class MLPDropout(Layer):
    def __init__(self,Dropout,num_layers,dim,out_dim,activation='relu',name=None):
        super().__init__()
        self.layers = [
            Dropout(Dense(dim,activation=activation, name = None if name is None else f'{name}_{l}'))
            for l in range(num_layers)
        ] + [Dropout(Dense(out_dim, name = None if name is None else f'{name}_{num_layers}'))]

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o
