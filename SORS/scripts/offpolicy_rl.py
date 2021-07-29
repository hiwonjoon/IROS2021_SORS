import os
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import gin

import gym
from SORS.modules.replay_buffer import ReplayBuffer
from SORS.modules.preference import PreferenceDataset
from SORS.modules.utils import setup_logger, tqdm, write_gin_config
from SORS.modules import env_utils

@gin.configurable(module=__name__)
def run(
    args,
    log_dir,
    seed,
    ############ Gin Configurable
    env_id,
    Algo,
    # Replay Buffer
    batch_size:int,
    min_buffer_size:int,
    # Training loop
    until:int, # maximum number of experience with env
    update_period:int,
    update_num:int,
    update_log_period:int,
    save_period,
    run_period, # generate a single trajectory
    eval_period, # generate 100 trajectories
    eval_env_id = None, # use env_id
    eval_policies = ['pi'],
    save_trajs = False, # Whether you want to save all experienced trajectories for debugging purpose?
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    chkpt_dir = Path(log_dir)/'chkpt'
    chkpt_dir.mkdir(parents=True,exist_ok=True)

    # Define Environment & Replay buffer
    env = gym.make(env_id)
    env.seed(seed)

    replay_buffer = ReplayBuffer()
    if save_trajs:
        trajs_buffer = PreferenceDataset(10_000,10_000,tfrecords_dir=os.path.join(log_dir,'preference_dataset'))

    # Define algorithm
    algo = Algo()
    update, report = algo.prepare_update()

    interact = env_utils.interact(env,algo.pi)

    # Define evaluation
    tests = {}
    for test_policy in eval_policies:
        test_env = gym.make(env_id if eval_env_id is None else eval_env_id)
        test_env.seed(seed)

        test_pi = getattr(algo,test_policy)
        test_interact = env_utils.interact(test_env, test_pi, stochastic=False)

        tests[test_policy] = test_interact

    def eval_policy(u,num_trajs):
        for pi_name, test_interact in tests.items():

            returns = []
            for _ in range(num_trajs):
                traj = env_utils.unroll_till_end(test_interact)
                returns.append(np.sum(traj.rewards))

                summary_writer.info('raw',f't/test_{pi_name}_eps_length',len(traj.states),u)
                summary_writer.info('raw',f't/test_{pi_name}_eps_return',np.sum(traj.rewards),u)
                try:
                    summary_writer.info('raw',f't/test_{pi_name}_eps_norm_return',test_env.get_normalized_score(np.sum(traj.rewards)),u)
                except:
                    pass

            summary_writer.info('raw',f't/test_{pi_name}_mean_eps_return',np.mean(returns),u)
            try:
                summary_writer.info('raw',f't/test_{pi_name}_mean_eps_norm_return',test_env.get_normalized_score(np.mean(returns)),u)
            except:
                pass

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        u = 0
        current_eps = []
        for t in tqdm(range(until),smoothing=0.):
            transition_tuple, last = next(interact)

            replay_buffer.append(transition_tuple)
            current_eps.append(transition_tuple)

            if last:
                traj, current_eps = env_utils.list_of_tuple_to_traj(current_eps), []
                summary_writer.info('raw','t/eps_length',len(traj.states),t)
                summary_writer.info('raw','t/eps_return',np.sum(traj.rewards),t)

                if save_trajs: trajs_buffer.add_traj(traj)

            if len(replay_buffer) < min_buffer_size:
                continue

            # Update
            if (t+1) % update_period == 0:
                for _ in range(update_num):
                    u += 1
                    s,a,r,ś,done,_ = replay_buffer.sample(batch_size)

                    update(s,a,r,ś,done)

                    # Update Log
                    if u % update_log_period == 0:
                        for name,item in report.items():
                            val = item.result().numpy()
                            summary_writer.info('raw',f'offpolicy_rl/{name}',val,u)
                            item.reset_states()

            # eval
            if (t+1) % eval_period != 0 and (t+1) % run_period == 0:
                eval_policy(t+1,num_trajs=1)

            if (t+1) % eval_period == 0:
                eval_policy(t+1,num_trajs=100)

            # save
            if (t+1) % save_period == 0:
                algo.save_weights(str(chkpt_dir),t+1)

        eval_policy(t+1,num_trajs=100)

    except KeyboardInterrupt:
        pass

    algo.save_weights(log_dir,with_Q=True)
    trajs_buffer.flush()

    logger.info('-------Gracefully finalized--------')
    logger.info('-------Bye Bye--------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file',required=True, nargs='+')
    parser.add_argument('--config_params', nargs='*', default='')

    args = parser.parse_args()

    config_params = '\n'.join(args.config_params)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    gin.parse_config_files_and_bindings(args.config_file, config_params)

    import SORS.scripts.offpolicy_rl
    SORS.scripts.offpolicy_rl.run(args,**vars(args))
