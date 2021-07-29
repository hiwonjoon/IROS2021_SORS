import gin
import gym

@gin.configurable(module=__name__)
class DelayedRewardEnv(gym.Wrapper):
    def __init__(
        self,
        env_id,
        reward_freq,
    ):
        env = gym.make(env_id)
        self.reward_freq = reward_freq

        super().__init__(env)

    def reset(self, **kwargs):
        self.t = 0
        self.delayed_reward = 0.

        return super().reset(**kwargs)

    def step(self, action):
        self.t += 1
        observation, orig_reward, done, info = super().step(action)
        info.update({'t':self.t, 'orig_reward':orig_reward})

        self.delayed_reward += orig_reward

        if self.t % self.reward_freq == 0 or done:
            reward = self.delayed_reward
            self.delayed_reward = 0.
        else:
            reward = 0

        return observation, reward, done, info
