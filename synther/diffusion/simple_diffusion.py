from typing import Tuple

import gin
import gymnasium as gym
import numpy as np

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import split_diffusion_samples



@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals
