"""
Soft-Actor-Critic
"""
import os
import logging

import gin
import numpy as np
import tensorflow as tf

from SORS.modules.value_function import ActionValue
from SORS.modules.stochastic_policy import StochasticPolicy
import SORS.modules.optimizer

@gin.configurable(module=__name__)
class SAC():
    def __init__(
        self,
    ):
        self.Q1 = ActionValue(name='Q1')
        self.Q2 = ActionValue(name='Q2')
        self.pi = StochasticPolicy()

    @gin.configurable(module=__name__)
    def prepare_update(
        self,
        gamma:float,
        entropy_reg_coeff:float, # alpha in SAC
        polyak_coeff:float,
        Optimizer=SORS.modules.optimizer.AdamOptimizer,
        tune_entropy_reg_coeff=False,
        target_entropy=None,
        min_entropy_reg_coeff=1e-3,
        max_entropy_reg_coeff=1e3,
    ):
        q_optimizer = Optimizer(
            self.Q1.trainable_variables + self.Q2.trainable_variables,
            self.Q1.trainable_decay_variables + self.Q2.trainable_decay_variables
        )

        if tune_entropy_reg_coeff:
            log_entropy_reg_coeff = tf.Variable(
                tf.math.log(entropy_reg_coeff),
                trainable=True,
                constraint=SORS.modules.optimizer.ClipConstraint(min_value=tf.math.log(min_entropy_reg_coeff), max_value=tf.math.log(max_entropy_reg_coeff)),
                dtype=tf.float32
            )
        else:
            log_entropy_reg_coeff = tf.math.log(entropy_reg_coeff)

        pi_update, pi_report = self.pi.prepare_update([self.Q1,self.Q2],log_entropy_reg_coeff,target_entropy)

        report = {}
        for key,item in pi_report.items(): report[f'pi/{key}'] = item
        for key,item in q_optimizer.report.items(): report[f'q/{key}'] = item

        @tf.function
        def update(s,a,r,ns,done):
            best_â, logp_best_â = self.pi.action(ns) # assuming the current poliyc is the best w.r.t given Q
            q_target = r + gamma*(1. - done)*(tf.math.minimum(self.Q1(ns,best_â,use_target=True),self.Q2(ns,best_â,use_target=True)) - tf.math.exp(log_entropy_reg_coeff) * logp_best_â)

            with tf.GradientTape() as tape:
                loss = 0.
                for Q in [self.Q1, self.Q2]:
                    q = Q(s,a,use_target=False)
                    loss += tf.reduce_mean(0.5 * (q - q_target)**2,axis=0)

            q_optimizer.minimize(tape,loss)

            pi_update(s)

            self.Q1.update_target(polyak_coeff)
            self.Q2.update_target(polyak_coeff)

        return update, report

    def save_weights(self,log_dir,it=None,with_Q=False):
        self.pi.save_weights(os.path.join(log_dir,'pi.tf' if it is None else f'pi-{it}.tf'))
        if with_Q:
            self.Q1.save_weights(os.path.join(log_dir,'Q1.tf' if it is None else f'Q1-{it}.tf'))
            self.Q2.save_weights(os.path.join(log_dir,'Q2.tf' if it is None else f'Q2-{it}.tf'))

    def load_weights(self,log_dir,it=None,with_Q=False):
        self.pi.load_weights(os.path.join(log_dir,'pi.tf' if it is None else f'pi-{it}.tf'))
        if with_Q:
            self.Q1.load_weights(os.path.join(log_dir,'Q1.tf' if it is None else f'Q1-{it}.tf'))
            self.Q2.load_weights(os.path.join(log_dir,'Q2.tf' if it is None else f'Q2-{it}.tf'))
