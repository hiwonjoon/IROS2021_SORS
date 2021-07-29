import gin
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from functools import partial
import tensorflow_probability as tfp
tfd = tfp.distributions

@gin.configurable(module=__name__)
def AdamOptimizer(beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False):
    return partial(tf.keras.optimizers.Adam,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon, amsgrad=amsgrad)

@gin.configurable(module=__name__)
def SGDOptimizer(momentum=0.9, nesterov=False):
    return partial(tf.keras.optimizers.SGD,momentum=momentum,nesterov=nesterov)

@gin.configurable(module=__name__)
class StochasticPolicy(Model):
    def __init__(self,Net,squash_output=True,original_std_model=True,scale=1.):
        super().__init__()

        self.net = Net()

        self.squash_output = squash_output
        self.scale = scale
        self.original_std_model = original_std_model

    def _action(self,ob,stochastic=True):
        o = self.net(ob)

        mu, sigma = tf.split(o,2,axis=-1)
        if self.original_std_model:
            LOG_STD_MAX = 2
            LOG_STD_MIN = -20
            sigma = tf.exp(tf.clip_by_value(sigma, LOG_STD_MIN, LOG_STD_MAX))
        else:
            sigma = tf.nn.softplus(tf.maximum(-15.,sigma))

        dist_u = tfd.MultivariateNormalDiag(mu,scale_diag=sigma,validate_args=False,allow_nan_stats=False)

        u = dist_u.sample() if stochastic else mu

        logp_u = dist_u.log_prob(u)

        if self.squash_output:
            a = tf.nn.tanh(u)
            # u ~ N(mu,sigma), a = tanh(u)
            # change of variables formula can be applied since tanh is invertible
            # borrowed from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L50
            logp_a = logp_u - tf.reduce_sum(2*(tf.math.log(2.) - u - tf.nn.softplus(-2*u)), axis=-1)
        else:
            a, logp_a  = u, logp_u

        return dist_u, (self.scale * a, logp_a)

    @tf.function
    def action(self,ob,stochastic=True):
        dist_u, (a, logp_a)= self._action(ob,stochastic)
        return a, logp_a

    @tf.function
    def action_sample(self,ob,num_samples):
        # Note that num_samples are PREpended, not appended
        dist_u, _= self._action(ob)
        return self.scale * dist_u.sample(num_samples)

    def __call__(self,ob,stochastic=True):
        if ob.ndim == 1:
            ob = ob[None]
            flatten = True
        else:
            flatten = False

        a, logp_a = self.action(ob,stochastic)

        if flatten:
            a, logp_a = a[0].numpy(), logp_a[0].numpy()

        return a, logp_a

    @gin.configurable(module=__name__)
    def get_gradients(
        self,
        Qs,
        ob,
        alpha,
        use_target_Q_for_optimize=False,
        max_grad_norm=0,
    ):
        with tf.GradientTape() as tape:
            dist, (a, log_p_a)= self._action(ob)

            entropy = tf.reduce_mean(dist.entropy())
            unconditional_entropy = -1. * tf.reduce_mean(log_p_a,axis=0)

            qs = tf.stack([
                Q(ob,a,use_target=use_target_Q_for_optimize) - alpha * log_p_a
                for Q in Qs],axis=-1)
            target = tf.reduce_min(qs,axis=-1)

            loss = -tf.reduce_mean(target,axis=0)

        gradients = tape.gradient(loss, self.net.trainable_variables)

        gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
        if max_grad_norm > 0:
            gradients = gradients_clipped

        return loss, (entropy, unconditional_entropy), gradients, grad_norm

    @gin.configurable(module=__name__)
    def prepare_update(
        self,
        Qs,
        log_alpha,
        target_entropy,
        learning_rate=1e-4,
        Optimizer=AdamOptimizer,
    ):
        optimizer = Optimizer()(learning_rate)

        reports = {
            'loss' : tf.keras.metrics.Mean(name='loss'),
            'grad_norm' : tf.keras.metrics.Mean(name='grad_norm'),
            'entropy' : tf.keras.metrics.Mean(name='entropy'),
            'unconditional_entropy' : tf.keras.metrics.Mean(name='unconditional_entropy'),
        }

        if isinstance(log_alpha,tf.Variable) and log_alpha.trainable:
            alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)
            reports['alpha'] = tf.keras.metrics.Mean(name='alpha')

            if target_entropy is None:
                target_entropy = -1. * gin.query_parameter('%ac_dim') #Follow heuristics given here; https://github.com/rail-berkeley/softlearning/blob/77538206e48343120f086e002d0105aab35e2a2b/softlearning/algorithms/sac.py#L39
        else:
            alpha_optimizer = None

        @tf.function
        def update_fn(ob):
            loss, (entropy, unconditional_entropy), gradients, grad_norm = self.get_gradients(Qs,ob,tf.math.exp(log_alpha))

            reports['entropy'](entropy)
            reports['unconditional_entropy'](unconditional_entropy)
            reports['loss'](loss)
            reports['grad_norm'](grad_norm)

            optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

            if alpha_optimizer is not None:
                with tf.GradientTape() as tape:
                    # We want entropy should be larger than target entropy
                    log_alpha_loss = -tf.math.exp(log_alpha) * (target_entropy - unconditional_entropy)

                grad = tape.gradient(log_alpha_loss,[log_alpha])
                alpha_optimizer.apply_gradients(zip(grad,[log_alpha]))

                reports['alpha'](tf.math.exp(log_alpha))

            return loss

        return update_fn, reports
