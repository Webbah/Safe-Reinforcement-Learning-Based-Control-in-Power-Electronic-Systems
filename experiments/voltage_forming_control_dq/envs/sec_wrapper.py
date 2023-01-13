from os import makedirs
from typing import Union

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0, Fastqueue
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn

from experiments.voltage_forming_control_dq.util.config import cfg
from feasible_set_test import feasible_set, A_d, B_d
from util.feasible_helper import calc_feasible_set
from util.safeguard import Safeguard



class SecWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 5000000,
                 n_trail="", integrator_weight=None, antiwindup_weight=None, gamma=0,
                 penalty_I_weight=1, penalty_P_weight=1, t_start_penalty_I=0, t_start_penalty_P=0,
                 number_learing_steps=500000, config=None, safe=0):
        """
        Env Wrapper to add features to the env-observations and adds information to env.step output which can be used in
        case of an continuing (non-episodic) task to reset the environment without being terminated by done

        Hint: is_dq0: if the control is done in dq0; if True, the action is tranfered to abc-system using env-phase and
            the observation is tranfered back to dq using the next phase

        :param env: Gym environment to wrap
        :param number_of_features: Number of features added to the env observations in the wrapped step method
        :param training_episode_length: (For non-episodic environments) number of training steps after the env is reset
            by the agent for training purpose (Set to inf in test env!)

        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.full(env.observation_space.shape[0] + number_of_features, -np.inf),
            high=np.full(env.observation_space.shape[0] + number_of_features, np.inf))

        self.training_episode_length = training_episode_length
        self._n_training_steps = 0
        self.n_episode = 0
        self.reward_episode_mean = []
        self.reward_plus_addon_episode_mean = []
        self.steps_per_episode = []
        self.terminated = 0
        self.n_trail = n_trail
        self.phase = []
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.integrator_weight = integrator_weight
        self.antiwindup_weight = antiwindup_weight
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)
        self.gamma = gamma
        self.penalty_I_weight = penalty_I_weight
        self.penalty_P_weight = penalty_P_weight
        self.t_start_penalty_I = t_start_penalty_I
        self.t_start_penalty_P = t_start_penalty_P
        self.number_learing_steps = number_learing_steps
        self.rew = []
        self.rew_sum = []
        self.penaltyP = []
        self.penaltyI = []
        self.clipped_rew = []
        self.cfg = config
        self.u_safe = []
        self.u_RL = []

        if cfg['loglevel'] == 'train':
            self.i_a = []
            self.i_b = []
            self.i_c = []
            self.v_a = []
            self.v_b = []
            self.v_c = []
            self.phase = []
            self.R_training = []
            self.u_a_RL = []
            self.u_b_RL = []
            self.u_c_RL = []
            self.u_a_safe = []
            self.u_b_safe = []
            self.u_c_safe = []
            self.safe_guard_failed = []
            self.safe_guard_rand_failed = []

        if safe:
            self.safe_guard = Safeguard(feasible_set, set_scaler=0, state_limits=np.array([300, 325 * 1.2]),
                                        action_limits=np.array([400]))

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """
        action_P = action[0:3]
        action_I = action[3:6]

        self.integrator_sum += action_I * self.integrator_weight

        action_PI = action_P + self.integrator_sum

        if self.cfg['is_dq0']:
            # Action: dq0 -> abc
            action_abc = dq0_to_abc(action_PI, self.env.net.components[0].phase)
        else:
            action_abc = action_PI

        """
        # Clipping not used - because safeguard is utilized
        # check if m_abc will be clipped
        if np.any(abs(action_abc) > 1):

            clipped_action = np.clip(action_abc, -1, 1)

            delta_action = clipped_action - action_abc
            # if, reduce integrator by clipped delta
            action_delta = abc_to_dq0(delta_action, self.env.net.components[0].phase)
            self.integrator_sum += action_delta * self.antiwindup_weight

            clip_reward = np.clip(np.sum(np.abs(delta_action) * \
                                         (-1 / (self.env.net.components[0].v_lim / self.env.net.components[
                                             0].v_DC / 2))) / 3,
                                  -1, 0) * (1 - self.gamma)

            # clip_reward = 0

            action_abc = clipped_action

        else:
            clip_reward = 0
        """

        sg_list = []

        obs_i_l = []
        obs_v_l = []
        a_safe = np.zeros(action_abc.shape[0])
        for a in range(3):
            #
            # asd = action_abc
            obs_i = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a]]
            obs_v = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a + 3]]
            action_safe, sg_active = self.safe_guard.guide(action_abc[a] * (self.env.env.net.components[0].v_DC / 2),
                                                           (obs_i, obs_v))
            # a_list.append(action_safe)
            a_safe[a] = action_safe / (self.env.env.net.components[0].v_DC / 2)
            sg_list.append(sg_active)
            obs_v_l.append(obs_v)
            obs_i_l.append(obs_i)
        self.u_safe.append(a_safe.tolist())
        self.u_RL.append(action_abc.tolist())

        # obs, reward, done, info = super().step(action_abc)
        obs, reward, done, info = super().step(a_safe)

        # reward = reward + clip_reward shifted to reward sum

        super().render()

        integrator_penalty = np.sum(-((np.abs(action_I)) ** 0.5)) / 3  # * (1 - self.gamma) / 3
        # action_P_penalty = - np.sum((np.abs(action_P - self.used_P)) ** 0.5) * (1 - self.gamma) / 3
        action_P_penalty = np.sum(-((np.abs(action_P)) ** 0.5)) / 3  # * (1 - self.gamma) / 3

        # reward_weight is = 1

        if self.total_steps > self.t_start_penalty_I:
            penalty_I_weight_scale = 1 / (self.t_start_penalty_I - self.number_learing_steps) * self.total_steps - \
                                     self.number_learing_steps / (self.t_start_penalty_I - self.number_learing_steps)

        else:
            penalty_I_weight_scale = 1

        if self.total_steps > self.t_start_penalty_P:
            penalty_P_weight_scale = 1 / (self.t_start_penalty_P - self.number_learing_steps) * self.total_steps - \
                                     self.number_learing_steps / (self.t_start_penalty_P - self.number_learing_steps)

        else:

            penalty_P_weight_scale = 1

        lam_P = self.penalty_P_weight * penalty_P_weight_scale
        lam_I = self.penalty_I_weight * penalty_I_weight_scale

        if reward > -1:
            # if reward = -1, env is abort, worst reward = -1, if not, sum up components:
            # reward_sum = (reward + clip_reward + lam_I * integrator_penalty + lam_P * action_P_penalty)
            reward_sum = (reward + lam_I * integrator_penalty + lam_P * action_P_penalty)

            # normalize r_sum between [-1, 1] from [-1-lam_P-lam_I, 1] using min-max normalization from
            # https://en.wikipedia.org/wiki/Feature_scaling

            # reward = 2 * (reward_sum + 1 + lam_P + lam_I) / (1 + 1 + lam_P + lam_I) - 1    # if clipped reward

            # without i:
            # reward = 1.5 * (reward_sum + lam_P + lam_I) / (1 + lam_P + lam_I) - 0.5

            # with i:
            reward = (1.75 * (reward_sum + lam_P + lam_I + 0.15) / (1 + lam_P + lam_I + 0.15) - 0.75) * (1 - self.gamma)

            if any(sg_list):
                # if safeguard was in action we give minimal possible reward defined by design to -0.5 * (1-gamma)
                reward = -0.75 * (1 - self.gamma)

        else:
            self.terminated = 1

        self._n_training_steps += 1

        # if self._n_training_steps % round(self.training_episode_length / 10) == 0:
        #    self.env.on_episode_reset_callback()

        if cfg[
            'loglevel'] == 'train':  # and (not self.n_episode % 20 or self.terminated):  # da terminated not known before.... only for debug
            self.R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])

            self.i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
            self.i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
            self.i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

            self.v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
            self.v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
            self.v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])
            self.phase.append(self.env.net.components[0].phase)

            self.u_a_safe.append(a_safe[0])
            self.u_b_safe.append(a_safe[1])
            self.u_c_safe.append(a_safe[2])
            self.u_a_RL.append(action_abc[0])
            self.u_b_RL.append(action_abc[1])
            self.u_c_RL.append(action_abc[2])

            self.safe_guard_failed.append(self.safe_guard.sol_not_found)
            self.safe_guard.sol_not_found = 0

        if self._n_training_steps % self.training_episode_length == 0:
            done = True
            super().close()

        """
        Features
        """
        error = (obs[6:9] - obs[3:6]) / 2  # control error: v_setpoint - v_mess
        # delta_i_lim_i_phasor = 1 - self.i_phasor  # delta to current limit

        """
        Following maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (phasor_mag) - 0.5. 0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)
        """
        # obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        # obs = np.append(obs, np.sin(self.env.net.components[0].phase))
        # obs = np.append(obs, np.cos(self.env.net.components[0].phase))

        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)

        # todo efficiency?
        self.used_P = np.copy(action_P)
        self.used_I = np.copy(self.integrator_sum)

        self.rew_sum.append(reward)

        if done:
            # log train curve with additional rewards:
            self.reward_plus_addon_episode_mean.append(np.mean(self.rew_sum))
            # log train curve with raw env-reward:
            self.reward_episode_mean.append(np.mean(self.rewards))
            self.steps_per_episode.append(self._n_training_steps)
            self.n_episode += 1

            if cfg['loglevel'] == 'train' and (not self.n_episode % 20 or self.terminated):
                episode_data = {
                    "R_load_training": self.R_training,
                    "i_a_training": self.i_a,
                    "i_b_training": self.i_b,
                    "i_c_training": self.i_c,
                    "v_a_training": self.v_a,
                    "v_b_training": self.v_b,
                    "v_c_training": self.v_c,
                    "u_a_RL": self.u_a_RL,
                    "u_b_RL": self.u_b_RL,
                    "u_c_RL": self.u_c_RL,
                    "u_a_safe": self.u_a_safe,
                    "u_b_safe": self.u_b_safe,
                    "u_c_safe": self.u_c_safe,
                    "Rewards_raw": self.rewards,
                    "Rewards_sum": self.rew_sum,
                    "Phase": self.phase,
                    "Solution_not_found": self.safe_guard_failed
                }

                n_trail = str(self.n_trail)

                save_folder = 'data/' + cfg['meas_data_folder']
                log_path = f'{cfg["STUDY_NAME"]}/{n_trail}/'

                df = pd.DataFrame(episode_data)
                makedirs(save_folder + log_path, exist_ok=True)
                df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                             'training_episode_number_' + str(self.n_episode) + '_terminated' + str(self.terminated) +
                             ".pkl.bz2")

                # clear lists
                self.R_training = []
                self.i_a = []
                self.i_b = []
                self.i_c = []
                self.v_a = []
                self.v_b = []
                self.v_c = []
                self.phase = []
                self.safe_guard_failed = []

            super().close()

            """
            plt.plot([item[0] for item in self.u_safe], 'b')
            plt.plot([item[1] for item in self.u_safe], 'r')
            plt.plot([item[2] for item in self.u_safe], 'g')
            plt.ylabel('u_safe')
            plt.grid()
            plt.show()

            plt.plot([item[0] for item in self.u_RL], 'b')
            plt.plot([item[1] for item in self.u_RL], 'r')
            plt.plot([item[2] for item in self.u_RL], 'g')
            plt.grid()
            plt.ylabel('u_RL')
            plt.show()

            self.u_RL.append([0, 0, 0])
            self.u_safe.append([0, 0, 0])
            episode_data = {
                "v_a": self.env.history.df.values[:, 1].tolist(),
                "i_a": self.env.history.df.values[:, 4].tolist(),
                "u_safe_a": [item[1] for item in self.u_safe],
                "u_RL_a": [item[1] for item in self.u_RL]
            }

            df = pd.DataFrame.from_dict(episode_data)
            #df.to_pickle(f"New2Episode_{str(self.n_episode)}.pkl.bz2")
            df.to_pickle(f"New2Episode_{str(self.n_episode)}.pkl.bz2")
            """

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """

        if cfg['loglevel'] == 'train':
            self.i_a = []
            self.i_b = []
            self.i_c = []
            self.v_a = []
            self.v_b = []
            self.v_c = []
            self.phase = []
            self.R_training = []
            self.u_a_RL = []
            self.u_b_RL = []
            self.u_c_RL = []
            self.u_a_safe = []
            self.u_b_safe = []
            self.u_c_safe = []
            self.safe_guard_failed = []

        obs = super().reset()

        self._n_training_steps = 0
        self.terminated = 0
        self.rew_sum = []
        self.rew = []
        self.integrator_sum = np.zeros(self.action_space.shape)
        self.used_P = np.zeros(self.action_space.shape)
        self.used_I = np.zeros(self.action_space.shape)

        """
        Features
        """
        error = (obs[6:9] - obs[3:6]) / 2  # control error: v_setpoint - v_mess
        # delta_i_lim_i_phasor = 1 - self.i_phasor  # delta to current limit

        """
        Following maps the return to the range of [-0.5, 0.5] in
        case of magnitude = [-lim, lim] using (phasor_mag) - 0.5. 0.5 can be exceeded in case of the magnitude
        exceeds the limit (no extra env interruption here!, all phases should be validated separately)
        """
        # obs = np.append(obs, self.i_phasor - 0.5)
        obs = np.append(obs, error)
        # obs = np.append(obs, np.sin(self.env.net.components[0].phase))
        # obs = np.append(obs, np.cos(self.env.net.components[0].phase))

        # obs = np.append(obs, delta_i_lim_i_phasor)
        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        obs = np.append(obs, self.used_I)
        # obs = np.append(obs, self.used_action)

        if cfg['loglevel'] == 'train':
            self.R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])

            self.i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
            self.i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
            self.i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

            self.v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
            self.v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
            self.v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])
            self.phase.append(self.env.net.components[0].phase)

            self.u_a_safe.append(0)
            self.u_b_safe.append(0)
            self.u_c_safe.append(0)
            self.u_a_RL.append(0)
            self.u_b_RL.append(0)
            self.u_c_RL.append(0)

            self.rewards.append(0)  # only for logging!!! Otherwise does alter the mean!!!
            self.rew_sum.append(0)

            self.safe_guard_failed.append(0)

        return obs


class SecWrapperPastVals(SecWrapper):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 500000,
                 n_trail="", integrator_weight=None, antiwindup_weight=None, gamma=0,
                 penalty_I_weight=1, penalty_P_weight=1, t_start_penalty_I=0, t_start_penalty_P=0,
                 number_learing_steps=500000, number_past_vals=10, config=None, safe=0):
        """
        Env Wrapper which adds the number_past_vals voltage ([3:6]!!!) observations to the observations.
        Initialized with zeros!
        """
        super().__init__(env, number_of_features, training_episode_length,
                         n_trail, integrator_weight, antiwindup_weight, gamma,
                         penalty_I_weight, penalty_P_weight, t_start_penalty_I, t_start_penalty_P,
                         number_learing_steps, config, safe)

        self.delay_queues = [Fastqueue(1, 3) for _ in range(number_past_vals)]

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        obs, reward, done, info = super().step(action)
        obs_delay_array = self.shift_and_append(obs[3:6])
        obs = np.append(obs, obs_delay_array)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset the wrapped env and the flag for the number of training steps after the env is reset
        by the agent for training purpose and internal counters
        """

        [x.clear() for x in self.delay_queues]
        obs = super().reset()
        obs_delay_array = self.shift_and_append(obs[3:6])
        obs = np.append(obs, obs_delay_array)

        return obs

    def shift_and_append(self, obs):
        """
        Takes the observation and shifts throught the queue
        every queue output is added to total obs
        """
        obs_delay_array = np.array([])
        obs_temp = obs
        for queue in self.delay_queues:
            obs_temp = queue.shift(obs_temp)
            obs_delay_array = np.append(obs_delay_array, obs_temp)

        return obs_delay_array