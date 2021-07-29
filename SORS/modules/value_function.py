import gin
import tensorflow as tf
from tensorflow.keras import Model

@gin.configurable(module=__name__)
class ActionValue(Model):
    def __init__(
        self,
        Net,
        build_target_net=False,
        name=None,
    ):
        super().__init__()

        self.net = Net(name=None if name is None else name)

        if build_target_net:
            self.target_net = Net(name=None if name is None else f'{name}_target')
            self.update_target(0.)
        else:
            self.target_net = self.net

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    @property
    def trainable_decay_variables(self):
        return self.net.decay_vars()

    @tf.function
    def __call__(self,ob,ac,use_target=True): # default is using target network
        if self.target_net and use_target:
            return tf.squeeze(self.target_net((ob,ac)),axis=-1)
        else:
            return tf.squeeze(self.net((ob,ac)),axis=-1)

    #@tf.function
    def update_target(self,τ):
        main_net_vars = sorted(self.net.trainable_variables,key = lambda v: v.name)
        target_net_vars = sorted(self.target_net.trainable_variables,key = lambda v: v.name)
        assert len(main_net_vars) > 0 and len(target_net_vars) > 0 and len(main_net_vars) == len(target_net_vars), f'{len(main_net_vars)} != {len(target_net_vars)}'

        for v_main,v_target in zip(main_net_vars,target_net_vars):
            v_target.assign(τ*v_target + (1-τ)*v_main)
