import numpy as np
import collections
from tqdm import tqdm

def _env_info(env):
    info = {}
    info['action_space'] = env.action_space
    info['observation_space'] = env.observation_space
    info['max_length'] = env.spec.max_episode_steps
    return info

def interact(env,policy,stochastic=True,render=False):
    env_info = _env_info(env)
    t, should_reset= np.inf, True
    while True:
        if should_reset:
            t, s, should_reset = 0, env.reset(), False

        a,_ = policy(s.astype(np.float32),stochastic)
        t,(ś,r,should_reset,_) = t+1, env.step(a)
        f = env.render('rgb_array') if render else None

        done = False
        if should_reset and t != env_info['max_length']:
            done = True # only set true when it actually dies in a episodic task.

        yield (s,a,r,ś,done,f), should_reset

        s = ś

Trajectory = collections.namedtuple('Trajectory', 'states actions rewards dones frames')

def list_of_tuple_to_traj(l):
    states, actions, rewards, next_states, dones, frames =\
        [np.array(elem) for elem in zip(*l)]

    states = np.concatenate([states,next_states[-1:]],axis=0)

    traj = Trajectory(
        states = states.astype(np.float32),
        actions = actions.astype(np.float32),
        rewards = rewards.astype(np.float32),
        dones = dones.astype(np.float32),
        frames = frames.astype(np.float32)
    )

    return traj

def unroll_till_end(interact,tqdm_disable=True):
    traj = []

    pbar = tqdm(disable=tqdm_disable)
    for transition_tuple, last in interact:
        pbar.update()

        traj.append(transition_tuple)
        if last: break
    pbar.close()

    traj = list_of_tuple_to_traj(traj)

    return traj

class RandomPolicy():
    def __init__(self,env,seed=None):
        self.env_info = _env_info(env)

        self.rg = np.random.RandomState(seed)
        self.ac_dim = self.env_info['action_space'].shape[-1]

        self.scale = self.env_info['action_space'].high[0]

        assert np.all(self.scale == self.env_info['action_space'].high) and \
            np.all(self.scale == -1.*self.env_info['action_space'].low)

    def seed(self,seed=None):
        self.rg = np.random.RandomState(seed)

    def __call__(self,ob,*args,**kwargs):
        if ob.ndim == 1:
            return self.rg.uniform(-self.scale,self.scale,[self.ac_dim]), None
        else:
            return self.rg.uniform(-self.scale,self.scale,[len(ob),self.ac_dim]), None

