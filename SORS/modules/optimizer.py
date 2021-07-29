import gin
import tensorflow as tf

class Optimizer():
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm,
    ):
        self.vars = vars
        self.decay_vars = decay_vars
        self.max_grad_norm = max_grad_norm

        self.report = {
            'loss': tf.keras.metrics.Mean(name='loss'),
        }
        if max_grad_norm > 0:
            self.report['grad_norm'] = tf.keras.metrics.Mean(name='grad_norm')

        self.optimizer = None

    def minimize(self,tape,loss):
        self.report['loss'](loss)
        gradients = tape.gradient(loss, self.vars)

        if self.max_grad_norm > 0:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.report['grad_norm'](grad_norm)

        self.optimizer.apply_gradients(zip(gradients, self.vars))

@gin.configurable(module=__name__)
class AdamOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False
    ):
        super().__init__(vars,decay_vars,max_grad_norm)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon, amsgrad=amsgrad)

@gin.configurable(module=__name__)
class SGDOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        lr=1e-3,
        momentum=0.9,
        nesterov=False
    ):
        super().__init__(vars,decay_vars,max_grad_norm)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=momentum,nesterov=nesterov)

class ClipConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be in some range"""

    def __init__(self, min_value, max_value):
        self.min_value, self.max_value = min_value, max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}
