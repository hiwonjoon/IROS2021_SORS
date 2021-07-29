import itertools
import tempfile
import atexit
from pathlib import Path
import shutil

import gin
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

@gin.configurable(module=__name__)
class PreferenceDataset(object):
    """
    Given a set of trajectory, provide a preference between every possible pair of trajectories
    """

    def __init__(
        self,
        max_in_memory_num, # how many trajectories kept in a memory?
        max_per_file_num, # how many trajectories kept in a single file?; (unit for the shuffle / flush)
        tfrecords_dir=None,
        use_state_and_action=False, # State-only (False) or, State-Action (True)
        ob_dim=None,
        ac_dim=None,
    ):
        self.max_in_memory_num = max_in_memory_num
        self.max_per_file_num = max_per_file_num

        self.tfrecords_dir = Path(tempfile.mkdtemp() if tfrecords_dir is None else tfrecords_dir)
        if tfrecords_dir is None:
            atexit.register(lambda : shutil.rmtree(str(self.tfrecords_dir)))

        self.remove_tfrecords_dir = tfrecords_dir is None

        self.trajs = []
        self.trajs_files = [str(f) for f in self.tfrecords_dir.glob('trajs.tfrecord-*')] # If there exist some preexisting dataset, then use it.

        self.use_state_and_action = use_state_and_action

        self.ob_dim = gin.query_parameter('%ob_dim') if ob_dim is None else ob_dim
        self.ac_dim = gin.query_parameter('%ac_dim') if ac_dim is None else ac_dim

    def flush(self):
        self.tfrecords_dir.mkdir(parents=True,exist_ok=True)

        for chunks in chunk(self.trajs,self.max_per_file_num):
            filename = str(self.tfrecords_dir / f'trajs.tfrecord-{len(self.trajs_files)}')
            with tf.io.TFRecordWriter(filename) as writer:
                for traj in chunks:
                    feature = {
                        'T': _int64_feature(len(traj.states)),
                        'states': _bytes_feature(tf.io.serialize_tensor(traj.states)),
                        'actions': _bytes_feature(tf.io.serialize_tensor(traj.actions)),
                        'rewards': _bytes_feature(tf.io.serialize_tensor(traj.rewards)),
                        'dones': _bytes_feature(tf.io.serialize_tensor(traj.dones)),
                        #'frames': _bytes_feature(tf.io.serialize_tensor(traj.frames)),
                    }
                    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(tf_example.SerializeToString())
            self.trajs_files.append(filename)
        self.trajs = []

    def add_traj(self,new_traj):
        self.trajs.append(new_traj)

        if len(self.trajs) >= self.max_in_memory_num:
            self.flush()

    def _sample_trajs(self):
        """
        Use both TF-records and from_tensor_slices for the best performance
        """
        traj_feature_description = {
            'T': tf.io.FixedLenFeature([],tf.int64),
            'states': tf.io.FixedLenFeature([],tf.string),
            'actions': tf.io.FixedLenFeature([],tf.string),
            'rewards': tf.io.FixedLenFeature([],tf.string),
            'dones': tf.io.FixedLenFeature([],tf.string),
            #'frames': tf.io.FixedLenFeature([],tf.string),
        }

        def _parse(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            features = tf.io.parse_single_example(example_proto, traj_feature_description)
            T = features['T']
            states = tf.io.parse_tensor(features['states'], out_type=tf.float32)
            actions = tf.io.parse_tensor(features['actions'], out_type=tf.float32)
            rewards = tf.io.parse_tensor(features['rewards'], out_type=tf.float32)
            dones = tf.io.parse_tensor(features['dones'], out_type=tf.float32)
            #frames = tf.io.parse_tensor(features['frames'], out_type=tf.float32)

            states.set_shape([None,self.ob_dim])
            actions.set_shape([None,self.ac_dim])
            rewards.set_shape([None])
            dones.set_shape([None])

            return states, actions, rewards, dones

        def _load(idxes):
            for idx in idxes:
                yield (self.trajs[idx].states,self.trajs[idx].actions,self.trajs[idx].rewards,self.trajs[idx].dones)

        in_memory_trajs = [list(chunks) for chunks in chunk(np.random.permutation(len(self.trajs)),self.max_per_file_num)]

        trajs_files = ['']*len(in_memory_trajs) + self.trajs_files
        trajs_idxes = tf.ragged.constant(in_memory_trajs + [[-1]]*len(self.trajs_files), inner_shape=())

        def _shard(f,idxes):
            if f == '':
                dataset = tf.data.Dataset.from_generator(
                    _load,
                    args=[idxes],
                    output_types=(tf.float32,tf.float32,tf.float32,tf.float32),
                    output_shapes=([None,self.ob_dim],[None,self.ac_dim],[None],[None])
                )
                dataset = dataset.shuffle(1,reshuffle_each_iteration=False) # Dataset type has to be equal to enable interleave / Don't need to shuffle since index is already shuffled
                return dataset
            else:
                dataset = tf.data.TFRecordDataset(f)
                dataset = dataset.map(_parse,num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.shuffle(self.max_per_file_num,reshuffle_each_iteration=True)
                #dataset = dataset.cache() # if dataset can be loaded to a memory --> use just memory / if not, cache will cause memory overflow.

                return dataset

        filenames = tf.data.Dataset.from_tensor_slices((trajs_files,trajs_idxes))
        filenames = filenames.shuffle(len(trajs_files),reshuffle_each_iteration=True)
        trajs = filenames.interleave(
            _shard,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            block_length=1, # has to be 1. (exclude a data that compares within the group)
            cycle_length=16) # maximum level of parallelism. (larger than 1 is enough, higher tends to be better but 16 would be enough)

        return trajs

    @gin.configurable(module=__name__)
    def batch(self,batch_size):
        if self.use_state_and_action:
            def _build(ab_states, ab_actions, ab_rewards, ab_dones):
                return (
                    tf.concat([ab_states[0][:-1],ab_actions[0]],axis=-1),
                    tf.concat([ab_states[1][:-1],ab_actions[1]],axis=-1),
                    tf.cast(tf.reduce_sum(ab_rewards[0]) < tf.reduce_sum(ab_rewards[1]),tf.int32)
                )
        else:
            def _build(ab_states, ab_actions, ab_rewards, ab_dones):
                return (
                    ab_states[0],
                    ab_states[1],
                    tf.cast(tf.reduce_sum(ab_rewards[0]) < tf.reduce_sum(ab_rewards[1]),tf.int32)
                )

        trajs = self._sample_trajs()

        dataset = trajs.apply(tf.data.experimental.dense_to_ragged_batch(2,drop_remainder=True))
        dataset = dataset.map(_build)
        dataset = dataset.repeat() # repeat indefinietly
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def batch_naive_v2(self,batch_size):
        def sample():
            N = len(self.trajs)
            while True:
                a, b = np.random.choice(N,2,replace=False)
                x1, x2 = self.trajs[a].states, self.trajs[b].states
                R1, R2 = np.sum(self.trajs[a].rewards), np.sum(self.trajs[b].rewards)

                yield x1, x2, R1 < R2

        dataset = tf.data.Dataset.from_generator(
            sample,
            output_types=(tf.float32,tf.float32,tf.int32),
            output_shapes = ([None,ob_dim],[None,ob_dim],[])
        )
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

@gin.configurable(module=__name__)
class PreferenceDatasetEnsemble(object):
    """
    Ensemble of PreferenceDataset
    """

    def __init__(
        self,
        num_ensembles,
        tfrecords_dir=None,
    ):
        self.dataset = PreferenceDataset(tfrecords_dir=tfrecords_dir)
        self.num_ensembles = num_ensembles

    def add_traj(self,new_traj):
        self.dataset.add_traj(new_traj)

    def batch(self,batch_size):
        def _unzip(a,b,y):
            return tuple((a[i],b[i],y[i]) for i in range(self.num_ensembles))

        dataset = self.dataset.batch(batch_size)
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(self.num_ensembles))
        dataset = dataset.map(_unzip)

        return dataset

        # Previous impl. (memory overuse.)
        #return tf.data.Dataset.zip(tuple(self.dataset.batch(batch_size) for _ in range(self.num_ensembles)))

class PreferenceDatasetSnippet(PreferenceDataset):
    """
    It will subsample from given trajectories, and provide a label based on its undiscounted sum of rewards of that 'subsampled' part.
    """

    def __init__(
        self,
        maximum_snippet_length=50,
    ):
        #TODO:
        pass

######################
# Test Case.
#######################
if __name__ == "__main__":
    from tqdm import tqdm
    import gym

    from modules import env_utils

    env = gym.make('Hopper-v2')
    ob_dim = 11
    ac_dim = 3
    batch_size = 10

    config = f"""
    ob_dim = {ob_dim}
    ac_dim = {ac_dim}
"""

    gin.parse_config_files_and_bindings([], config)

    D = PreferenceDataset(100,10,use_state_and_action=True)

    pi = env_utils.RandomPolicy(env)
    interact = env_utils.interact(env,pi)
    for i in tqdm(range(1000)):
        traj = env_utils.unroll_till_end(interact)

        D.add_traj(traj)

        if (i+1) % 30 == 0:
            it = iter(D.batch(batch_size))
            #it = iter(D.batch_naive_v2(batch_size))

            for _ in range(1000):
                (A,B,l) = next(it)
                tqdm.write(f'{A[1].shape}, {B[1].shape}, {l}')
