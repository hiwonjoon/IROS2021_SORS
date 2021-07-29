from gym.envs.registration import register
from SORS.envs.delayed_rewards import DelayedRewardEnv

register(
    id='delayed-reward-v0',
    entry_point='SORS.envs:DelayedRewardEnv',
    kwargs={},
    max_episode_steps=1000,
)
