from typing import Tuple, Optional, List, Type

import gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import TD3

# based on https://github.com/DLR-RM/stable-baselines3/issues/425
# UPDATE: since sb3 >1.6.X no register policy anymore. More information in
# https://github.com/DLR-RM/stable-baselines3/issues/1126
from stable_baselines3.common.policies import BaseModel, register_policy  # , register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import Actor, TD3Policy

"""
File to generate a custom policy based on https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
"""

def mlp(sizes, activation, output_activation, alpha_relu):
    """
    Defines a multi layer perceptron using pytorch layers and activation funtions
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act(negative_slope=alpha_relu)]
        else:
            if act is not None:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
            else:
                layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """

    def __init__(self, alpha_relu_actor, weight_scale, bias_scale, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        # self.mu = nn.Sequential(...)
        pi_sizes = [kwargs['observation_space'].shape[0]] + kwargs['net_arch'] + [kwargs['action_space'].shape[0]]
        self.mu = mlp(pi_sizes, kwargs['activation_fn'], nn.Tanh, alpha_relu_actor)

        count = 0
        # scale down weights and bias
        for kk in range(self.net_arch.__len__() + 1):
            self.mu._modules[str(count)].weight.data = self.mu._modules[
                                                                  str(count)].weight.data * weight_scale
            #self.actor_target.mu._modules[str(count)].weight.data = self.actor_target.mu._modules[
            #                                                             str(count)].weight.data * weight_scale

            self.mu._modules[str(count)].bias.data = self.mu._modules[str(count)].bias.data * bias_scale
            #self.actor_target.mu._modules[str(count)].bias.data = self.actor.mu._modules[
            #                                                           str(count)].bias.data * bias_scale

            count += 2

class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """

    def __init__(
            self,
            alpha_relu_critic,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            # q_net = nn.Sequential(...)
            input_space = [self.observation_space.shape[0] + + self.action_space.shape[0]]
            q_sizes = input_space + net_arch + [1]
            q_net = mlp(q_sizes, activation_fn, None, alpha_relu=alpha_relu_critic)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        self.alpha_relu_actor = kwargs.pop('alpha_relu_actor')
        self.alpha_relu_critic = kwargs.pop('alpha_relu_critic')
        self.weight_scale = kwargs.pop('weight_scale')
        self.bias_scale = kwargs.pop('bias_scale')
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(self.alpha_relu_actor, self.weight_scale, self.bias_scale, ** actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(self.alpha_relu_critic, **critic_kwargs).to(self.device)

class StaticEnv:
    def __init__(self, obs_length=1, action_length=1, obs_max=np.inf, obs_min=-np.inf, act_max=1, act_min=-1):
        self.observation_space = gym.spaces.Box(low=np.full(obs_length, obs_min), high=np.full(obs_length, obs_max))
        self.action_space = gym.spaces.Box(low=np.full(action_length, act_min), high=np.full(action_length, act_max))
        self.metadata = {} # render mode,... needed for definition purpose

register_policy("CustomTD3Policy", CustomTD3Policy)
#TD3.policy_aliases["CustomTD3Policy"] = CustomTD3Policy