import gin
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Model

@gin.configurable(module=__name__)
class RewardV2(Model):
    """
    Separate phi and reward_weight
    """

    def __init__(
        self,
        phi_dim,
        Phi,
        use_state_and_action=False, # State-only (False) or State-Action (True) for phi input
        use_bias=False,
    ):
        super().__init__()

        ob_dim = gin.query_parameter('%ob_dim')
        ac_dim = gin.query_parameter('%ac_dim')

        self.use_state_and_action = use_state_and_action
        self.use_bias = use_bias

        if self.use_state_and_action:
            in_dim = ob_dim + ac_dim
        else:
            in_dim = ob_dim

        if self.use_bias:
            self.phi_dim = phi_dim + 1
        else:
            self.phi_dim = phi_dim

        w = tf.keras.layers.Dense(1,use_bias=False)
        w.build((self.phi_dim,))

        self.w_net = tf.keras.layers.TimeDistributed(w)
        self.phi_net = Phi(in_dim=in_dim,out_dim=phi_dim)

    @tf.function
    def R(self,x): #return, reserved word...
        """
        inp:
            x: Ragged Tensor [B,T(None;ragged),feature_dim]
        out:
            R: tf.Tensor shape of [B]
        """
        if self.use_bias:
            ones = tf.RaggedTensor.from_row_lengths(
                values=tf.ones((tf.reduce_sum(x.row_lengths()),1),tf.float32),
                row_lengths=x.row_lengths()
            )
            phi = tf.concat([self.phi_net(x), ones],axis=-1)
            return tf.squeeze(tf.reduce_sum(self.w_net(phi),axis=1),axis=-1)
        else:
            return tf.squeeze(tf.reduce_sum(self.w_net(self.phi_net(x)),axis=1),axis=-1)

    @gin.configurable(module=f'{__name__}.RewardV2')
    def prepare_update(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
    ):
        optimizer = tfa.optimizers.AdamW(weight_decay,learning_rate)
        reports = {
            'loss':tf.keras.metrics.Mean(),
            'wd_loss':tf.keras.metrics.Mean(),
        }
        decay_vars = self.phi_net.decay_vars() + [self.w_net.layer.kernel]

        @tf.function
        def update_fn(b):
            """
            y = 0 if x1 > x2 (x1 is better trajectory) 1 otherwise.
            """
            x1,x2,y = b
            with tf.GradientTape() as tape:
                v1 = self.R(x1)
                v2 = self.R(x2)

                logits = tf.stack([v1,v2],axis=1) #[B,2]
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)

                loss = tf.reduce_mean(loss)
                reports['loss'](loss)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables),decay_var_list=decay_vars)

            wd = tf.add_n([tf.reduce_sum(var**2) for var in decay_vars])
            reports['wd_loss'](wd)

            return loss

        return update_fn, reports

    ################
    # Use in the external call
    ################
    @tf.function
    def call(self,s,a):
        """
        inp:
            s: Tensor [B,state_dim]
            a: Tensor [B,action_dim]
        out:
            R: tf.Tensor shape of [B]
        """
        return tf.squeeze(tf.matmul(self.phi(s,a),self.w()),axis=-1)

    #############
    # For Successor Feature Learning.
    #############
    @tf.function
    def phi(self,s,a):
        """
        inp:
            s: Tensor [B,state_dim]
            a: Tensor [B,action_dim]
        out:
            phi(x): tf.Tensor shape of phi(x) [B,phi_dim]
        """

        if self.use_state_and_action:
            phi = tf.squeeze(self.phi_net(tf.concat([s,a],axis=-1)[:,None,:]),axis=1)
        else:
            phi = tf.squeeze(self.phi_net(s[:,None,:]),axis=1)

        if self.use_bias:
            phi = tf.concat([phi, tf.expand_dims(tf.ones(tf.shape(phi)[:1]),axis=-1)],axis=-1)

        return phi

    @gin.configurable(module=f'{__name__}.RewardV2')
    def w(self,normalize=False):
        """
        out:
            weight vector: tf.Tensor shape of [phi_dim,1]
        """
        if normalize:
            return tf.linalg.normalize(self.w_net.layer.kernel)[0]
        else:
            return self.w_net.layer.kernel

@gin.configurable(module=__name__)
class RewardV2Ensemble(Model):
    """
    Ensemble of Reward V2
    """

    def __init__(
        self,
        num_ensembles
    ):
        super().__init__()

        self.ensembles = [RewardV2() for _ in range(num_ensembles)]
        self.phi_dim = self.ensembles[0].phi_dim * num_ensembles

    @tf.function
    def R(self,x): #return, reserved word...
        """
        inp:
            x: Ragged Tensor [B,T(None;ragged),feature_dim]
        out:
            R: tf.Tensor shape of [B]
        """
        return tf.add_n([reward.R(x) for reward in self.ensembles]) / len(self.ensembles)

    def prepare_update(self):
        agg_reports = {
            'loss':tf.keras.metrics.Mean(),
            'wd_loss':tf.keras.metrics.Mean(),
        }

        update_fns, reports = zip(*[reward.prepare_update() for reward in self.ensembles])

        @tf.function
        def update_fn(e_data):
            """
            y = 0 if x1 > x2 (x1 is better trajectory) 1 otherwise.
            """
            losses = []
            for update_fn, report, b in zip(update_fns, reports, e_data):
                loss = update_fn(b)

                # Aggregate Logs
                for name, item in agg_reports.items():
                    item(report[name].result())
                    report[name].reset_states()

                losses.append(loss)

            return tf.add_n(losses)

        return update_fn, agg_reports

    ################
    # Use in the external call
    ################
    @tf.function
    def call(self,s,a):
        """
        inp:
            s: Tensor [B,state_dim]
            a: Tensor [B,action_dim]
        out:
            R: tf.Tensor shape of [B]
        """
        return tf.add_n([reward(s,a) for reward in self.ensembles]) / len(self.ensembles)

    #############
    # For Successor Feature Learning.
    #############
    @tf.function
    def phi(self,s,a):
        """
        inp:
            s: Tensor [B,state_dim]
            a: Tensor [B,action_dim]
        out:
            phi(x): tf.Tensor shape of phi(x) [B,phi_dim]
        """
        return tf.concat([reward.phi(s,a) for reward in self.ensembles],axis=-1)

    def w(self):
        """
        out:
            weight vector: tf.Tensor shape of [phi_dim,1]
        """
        return tf.concat([reward.w() for reward in self.ensembles],axis=0)
