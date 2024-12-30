import d4rl
import gin
import gymnasium as gym
import numpy as np

# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs

