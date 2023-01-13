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
from feasible_set_test import feasible_set, A_d, B_d, W_x, omega_x, W_u, omega_u, i_nom, i_dead, v_lim, v_dead
from util.feasible_helper import calc_feasible_set
from util.rls_model import RLSFit
from util.safeguard import Safeguard



class BaseWrapper(Monitor):

    def __init__(self, env, number_of_features: int = 0, training_episode_length: int = 5000000,
                  n_trail="", gamma=0,
                 number_learing_steps=500000, number_past_vals=0, config=None, safe=1, learn_model=0, SG=None,
                 RLS_model=None):
        """
        Base Env Wrapper to add features to the env-observations and adds information to env.step output which can be
        used in case of an continuing (non-episodic) task to reset the environment without being terminated by done

        Hint: is_dq0 from the config: if the control is done in dq0; if True, the action is tranfered to abc-system using env-phase and
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

        # increase action-space for PI-seperation
        # self.action_space=gym.spaces.Box(low=np.full(d_i, -1), high=np.full(d_i, 1))

        self.training_episode_length = training_episode_length

        self._n_training_steps = 0
        self.n_episode = 0
        self.log_name_added = ''
        self.reward_episode_mean = []
        self.n_trail = n_trail
        self.phase = []
        self.used_P = np.zeros(self.action_space.shape)
        self.gamma = gamma
        self.number_learing_steps = number_learing_steps
        self.delay_queues = [Fastqueue(1, 3) for _ in range(number_past_vals)]
        self.cfg = config
        self.rew_sum = []
        self.reward_episode_mean = []
        self.reward_plus_addon_episode_mean = []
        self.steps_per_episode = []
        self.terminated = 0

        self.deterministic_run = False

        self.A_error_mean_per_episode = []
        self.B_error_mean_per_episode = []
        self.obs_hat_error_mea_per_episode = []

        self.u_safe = []
        self.u_RL = []
        self.A_error = []
        self.B_error = []

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
            self.safe_guard_active = []
            self.action_prev = np.zeros(self.action_space.shape)
            self.obs_hat_error = []
            self.obs_hat_error_i_a = []
            self.obs_hat_error_i_b = []
            self.obs_hat_error_i_c = []
            self.obs_hat_error_v_a = []
            self.obs_hat_error_v_b = []
            self.obs_hat_error_v_c = []
            self.obs_hat_error_all = []
            self.A11_a = []
            self.A11_b = []
            self.A11_c = []
            self.A12_a = []
            self.A12_b = []
            self.A12_c = []
            self.A21_a = []
            self.A21_b = []
            self.A21_c = []
            self.A22_a = []
            self.A22_b = []
            self.A22_c = []
            self.B1_a = []
            self.B1_b = []
            self.B1_c = []
            self.B2_a = []
            self.B2_b = []
            self.B2_c = []
            self.Poly_update_per_episode= []
            self.poly_update_count = 0

        if safe == 1:
            if learn_model:
                n, m = B_d.shape  # use the system dimension from the one we assumed
                self.guard_update_steps = 100
                self.model_delta = 0.00000000000001
                self.model_id = RLSFit(n, m, mu=0.99)
                feasible_init, A_hat, B_hat = self.init_model_fit()
                self.safe_guard = Safeguard(feasible_init, set_scaler=0, state_limits=np.array([i_nom/i_dead, 325 * v_lim/v_dead]),
                                            action_limits=np.array([1]), A_d=A_hat, B_d=B_hat,
                                            ts=self.env.env.time_step_size)
            else:
                self.model_id = None
                self.safe_guard = Safeguard(feasible_set, set_scaler=0, state_limits=np.array([i_nom/i_dead, v_lim/v_dead]),
                                        action_limits=np.array([1]), A_d=A_d, B_d=B_d, ts=self.env.env.time_step_size)

                #self.safe_guard2 = Safeguard(feasible_set2, set_scaler=0, state_limits=np.array([300, 325*1.2]),
                #                        action_limits=np.array([400]), A_d=A_d2, B_d=B_d2, ts=self.env.env.time_step_size)
        elif safe == 2:
            self.safe_guard = SG
            if learn_model:
                n, m = B_d.shape
                self.model_id = RLS_model
                self.guard_update_steps = 100
                self.model_delta = 0.00000000000001
                self.A_prev, self.B_prev = np.random.random((n, n)), np.random.random(
                    (n, m))
        else:
            self.safe_guard = None
            self.model_id = None



    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Adds additional features and infos after the gym env.step() function is executed.
        Triggers the env to reset without done=True every training_episode_length steps
        """

        if self.cfg['is_dq0']:
            # Action: dq0 -> abc
            if self.env.env.action_time_delay == 1:
                action_abc = dq0_to_abc(action, self.env.net.components[0].phase+2*np.pi*self.env.net.freq_nom*self.env.net.ts)
            else:
                action_abc = dq0_to_abc(action, self.env.net.components[0].phase)
        else:
            action_abc = action

        sg_list = []
        #a_abc = np.array([0.7, 0, 0])
        if self.safe_guard is not None:
            a_safe = np.zeros(action_abc.shape[0])
            for a in range(3):

                obs_i = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a]] / self.env.env.net.components[0].i_lim
                obs_v = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a + 3]]/ self.env.env.net.components[0].v_lim


                if self.env.env.action_time_delay == 1:
                    # use action of one step before to see how state will develop
                    # action_prev will be applied now
                    obs_i, obs_v = self.safe_guard.predict(np.array([obs_i, obs_v]), self.action_prev[a])
                                                           #*(self.env.env.net.components[0].v_DC/2))

                action_safe, sg_active = self.safe_guard.guide(action_abc[a], #* (self.env.env.net.components[0].v_DC/2),
                                                               (obs_i, obs_v))

                # a_list.append(action_safe)
                a_safe[a] = action_safe #/ (self.env.env.net.components[0].v_DC / 2)
                sg_list.append(sg_active)

            if self.env.env.action_time_delay == 1:
                self.action_prev = a_safe
                if self.cfg['is_dq0']:
                    a_safe_dq0 = abc_to_dq0(a_safe, self.env.net.components[0].phase+2*np.pi*self.env.net.freq_nom*self.env.net.ts)
                else:
                    a_safe_dq0 = a_safe
            else:
                if self.cfg['is_dq0']:
                    a_safe_dq0 = abc_to_dq0(a_safe, self.env.net.components[0].phase)
                else:
                    a_safe_dq0 = a_safe

            self.u_safe.append(a_safe.tolist())
            self.u_RL.append(action_abc.tolist())

            #
            obs, reward, done, info = super().step(a_safe)
        else:
            obs, reward, done, info = super().step(action_abc)



        if reward > -1:
            #reward = reward + clip_reward
            # scale reward between 1->-0.75 to compare with SEC
            #reward = (1.75 * (reward + 0.15) / (1 + 0.15) - 0.75) * (1 - self.gamma)
            reward = reward * (1 - self.gamma)

            if any(sg_list):
                # if safeguard was in action we give minimal possible reward defined by design to -0.5 * (1-gamma)
                reward = -0.75 * (1-self.gamma)
        else:
            self.terminated = 1


        ##### update model based on new observation
        # fit model on observations
        if self.model_id is not None:
            A_error_tmp = []
            B_error_tmp = []
            obs_hat_error_tmp = []
            obs_hat_error_tmp_i = []
            obs_hat_error_tmp_v = []

            for a in range(3):

                obs_i_next = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a]] / self.env.env.net.components[0].i_lim
                obs_v_next = self.env.history.df.values[-1, self.env.env._out_obs_tmpl._data[0][a + 3]] / self.env.env.net.components[0].v_lim

                obs_i = self.env.history.df.values[-2, self.env.env._out_obs_tmpl._data[0][a]] / self.env.env.net.components[0].i_lim
                obs_v = self.env.history.df.values[-2, self.env.env._out_obs_tmpl._data[0][a + 3]] / self.env.env.net.components[0].v_lim

                if self.env.env.action_time_delay == 1:
                    # consider Delay!? todo: correct?
                    action = self.used_P_abc[a] #* (self.env.env.net.components[0].v_DC / 2)
                else:
                    action = a_safe[a] #* (self.env.env.net.components[0].v_DC / 2)

                state = np.atleast_1d(np.squeeze((obs_i, obs_v)))
                next_state = np.atleast_1d(np.squeeze((obs_i_next, obs_v_next)))
                action = np.atleast_1d(np.squeeze(action))
                y_hat = self.model_id.predict((state, action))

                obs_hat_error_tmp.append(np.clip(np.max(np.abs(next_state - y_hat) / self.safe_guard.state_lim), 0, 1))
                obs_hat_error_tmp_i.append(np.abs(next_state[0] - y_hat[0]) )
                obs_hat_error_tmp_v.append(np.abs(next_state[1] - y_hat[1]) )

                                                # todo: or scaling based on limit? 2*!
                                                #np.array([self.env.env.net.components[0].i_lim,
                                                #          self.env.env.net.components[0].v_lim]))

                _, model_est = self.model_id.fit(
                    (state, action), next_state, return_mode='all')
                model_error = self.model_id.calc_error(metric='Error')
                # update system components
                # self.update(state, action, 'variance')
                P1 = self.model_id.estimator[0].R
                P2 = self.model_id.estimator[1].R
                rls_cov = [P1, P2]
                # calculate estimation-variance
                xi = np.concatenate([np.array([obs_i, obs_v]), action], axis=0)
                # variance for one predicted state
                var_y1 = xi.T @ P1 @ xi
                var_y2 = xi.T @ P2 @ xi
                var_estimate = (var_y1, var_y2)

                # update feasible set based on estimated model parameters
                self.update(model_est, var_estimate, obs_hat_error_tmp[a], a)#, episode=ep)


                A_hat, B_hat = model_est
                if a == 0:
                    self.A11_a.append(A_hat[0, 0])
                    self.A12_a.append(A_hat[0, 1])
                    self.A21_a.append(A_hat[1, 0])
                    self.A22_a.append(A_hat[1, 1])
                    self.B1_a.append(B_hat[0, 0])
                    self.B2_a.append(B_hat[1, 0])
                if a == 1:
                    self.A11_b.append(A_hat[0, 0])
                    self.A12_b.append(A_hat[0, 1])
                    self.A21_b.append(A_hat[1, 0])
                    self.A22_b.append(A_hat[1, 1])
                    self.B1_b.append(B_hat[0, 0])
                    self.B2_b.append(B_hat[1, 0])
                if a == 2:
                    self.A11_c.append(A_hat[0, 0])
                    self.A12_c.append(A_hat[0, 1])
                    self.A21_c.append(A_hat[1, 0])
                    self.A22_c.append(A_hat[1, 1])
                    self.B1_c.append(B_hat[0, 0])
                    self.B2_c.append(B_hat[1, 0])


                A_error_tmp.append(np.linalg.norm(
                    A_hat - A_d) / np.linalg.norm(
                    A_d))
                B_error_tmp.append(np.linalg.norm(
                    B_hat - B_d) / np.linalg.norm(
                    B_d))



            self.A_error.append(np.mean(A_error_tmp))
            self.B_error.append(np.mean(B_error_tmp))
            self.obs_hat_error.append(np.mean(obs_hat_error_tmp))
            self.obs_hat_error_i_a.append(obs_hat_error_tmp_i[0])
            self.obs_hat_error_i_b.append(obs_hat_error_tmp_i[1])
            self.obs_hat_error_i_c.append(obs_hat_error_tmp_i[2])
            self.obs_hat_error_v_a.append(obs_hat_error_tmp_v[0])
            self.obs_hat_error_v_b.append(obs_hat_error_tmp_v[1])
            self.obs_hat_error_v_c.append(obs_hat_error_tmp_v[2])

            self.safe_guard.obs_hat_error = np.mean(obs_hat_error_tmp)*20+1



        super().render()

        self.rew_sum.append(reward)

        self._n_training_steps += 1

        # if self._n_training_steps % round(self.training_episode_length / 10) == 0:
        #    self.env.on_episode_reset_callback()

        if cfg['loglevel'] == 'train':  # da terminated not known before.... only for debug
            self.R_training.append(self.env.history.df['r_load.resistor1.R'].iloc[-1])

            self.i_a.append(self.env.history.df['lc.inductor1.i'].iloc[-1])
            self.i_b.append(self.env.history.df['lc.inductor2.i'].iloc[-1])
            self.i_c.append(self.env.history.df['lc.inductor3.i'].iloc[-1])

            self.v_a.append(self.env.history.df['lc.capacitor1.v'].iloc[-1])
            self.v_b.append(self.env.history.df['lc.capacitor2.v'].iloc[-1])
            self.v_c.append(self.env.history.df['lc.capacitor3.v'].iloc[-1])
            self.phase.append(self.env.net.components[0].phase)


            self.u_a_RL.append(action_abc[0])
            self.u_b_RL.append(action_abc[1])
            self.u_c_RL.append(action_abc[2])

            if self.safe_guard is not None:
                self.u_a_safe.append(a_safe[0])
                self.u_b_safe.append(a_safe[1])
                self.u_c_safe.append(a_safe[2])
                self.safe_guard_failed.append(self.safe_guard.sol_not_found)
                self.safe_guard_rand_failed.append(self.safe_guard.rand_sol_not_found)
                self.safe_guard_active.append(int(any(sg_list)))
                self.safe_guard.sol_not_found = 0
                self.safe_guard.rand_sol_not_found = 0

        if self._n_training_steps % self.training_episode_length == 0:
            # info["timelimit_reached"] = True
            done = True
            super().close()

        if done:
            self.reward_episode_mean.append(np.mean(self.rewards))
            self.reward_plus_addon_episode_mean.append(np.mean(self.rew_sum))
            self.steps_per_episode.append(self._n_training_steps)
            self.A_error_mean_per_episode.append(np.mean(self.A_error[1:]))
            self.B_error_mean_per_episode.append(np.mean(self.B_error[1:]))
            self.obs_hat_error_mea_per_episode.append(np.mean(self.obs_hat_error[1:]))
            self.Poly_update_per_episode.append(self.poly_update_count)


            if cfg['loglevel'] == 'train':# and (not self.n_episode % 10 or self.terminated):  #todo: store teminated?
                if self.safe_guard is not None:
                    if self.model_id is not None:
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
                                    "A_error": self.A_error,
                                    "B_error": self.B_error,
                                    "obs_hat_error": self.obs_hat_error,
                                    "obs_hat_error_i_a": self.obs_hat_error_i_a,
                                    "obs_hat_error_i_b": self.obs_hat_error_i_b,
                                    "obs_hat_error_i_c": self.obs_hat_error_i_c,
                                    "obs_hat_error_v_a": self.obs_hat_error_v_a,
                                    "obs_hat_error_v_b": self.obs_hat_error_v_b,
                                    "obs_hat_error_v_c": self.obs_hat_error_v_c,
                                    "Rewards_raw": self.rewards,
                                    "Rewards_sum": self.rew_sum,
                                    "Phase": self.phase,
                                    "A11_a": self.A11_a,
                                    "A11_b": self.A11_b,
                                    "A11_c": self.A11_c,
                                    "A12_a": self.A12_a,
                                    "A12_b": self.A12_b,
                                    "A12_c": self.A12_c,
                                    "A21_a": self.A21_a,
                                    "A21_b": self.A21_b,
                                    "A21_c": self.A21_c,
                                    "A22_a": self.A22_a,
                                    "A22_b": self.A22_b,
                                    "A22_c": self.A22_c,
                                    "B1_a": self.B1_a,
                                    "B1_b": self.B1_b,
                                    "B1_c": self.B1_c,
                                    "B2_a": self.B2_a,
                                    "B2_b": self.B2_b,
                                    "B2_c": self.B2_c,
                                    "W_scale_a": self.W_scale_list_a,
                                    "Poly_update_a": self.Poly_update_a,
                                    "W_scale_b": self.W_scale_list_b,
                                    "Poly_update_b": self.Poly_update_b,
                                    "W_scale_c": self.W_scale_list_c,
                                    "Poly_update_c": self.Poly_update_c,
                                    "Solution_not_found": self.safe_guard_failed,
                                    "Rand_solution_not_found": self.safe_guard_rand_failed
                                   }
                        poly_vertices = {
                            "scaled_vertices": self.safe_guard.vertices
                        }
                    else:
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
                            "Solution_not_found": self.safe_guard_failed,
                            "Rand_solution_not_found": self.safe_guard_rand_failed,
                            "safe_guard_active": self.safe_guard_active
                        }
                else:
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
                        "Rewards_raw": self.rewards,
                        "Rewards_sum": self.rew_sum,
                        "Phase": self.phase,
                    }

                n_trail = str(self.n_trail)

                save_folder = 'data/' + cfg['meas_data_folder']
                log_path = f'{cfg["STUDY_NAME"]}/{n_trail}/'

                df = pd.DataFrame(episode_data)
                makedirs(save_folder + log_path, exist_ok=True)
                df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                             'training_episode_number_' + str(self.n_episode) + '_terminated' + str(self.terminated) +
                             self.log_name_added +".pkl.bz2")

                #df = pd.DataFrame(poly_vertices)
                #makedirs(save_folder + log_path, exist_ok=True)
                #df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                #             'vertices_episode_number_' + str(self.n_episode) + '_terminated' + str(self.terminated) +
                #             ".pkl.bz2")


            self.n_episode += 1

        """
        Features
        """
        if cfg['is_dq0']:
            # if setpoint in dq: Transform measurement to dq0!!!!
            obs[3:6] = abc_to_dq0(obs[3:6], self.env.net.components[0].phase)
            obs[0:3] = abc_to_dq0(obs[0:3], self.env.net.components[0].phase)

        error = (obs[6:9] - obs[3:6]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)
        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        #obs = np.append(obs, self.used_P2)
        obs_delay_array = self.shift_and_append(obs[3:6])
        obs = np.append(obs, obs_delay_array)

        # todo efficiency?
        if self.safe_guard is not None:
            # store the action which is given to env for next feature (delay of 1 only!)
            self.used_P_abc = np.copy(a_safe)
            self.used_P = np.copy(a_safe_dq0)
        else:
            self.used_P = np.copy(action)

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
            self.safe_guard_rand_failed = []
            self.safe_guard_active = []
            self.A_error = []
            self.B_error = []
            self.obs_hat_error = []
            self.obs_hat_error_i_a = []
            self.obs_hat_error_i_b = []
            self.obs_hat_error_i_c = []
            self.obs_hat_error_v_a = []
            self.obs_hat_error_v_b = []
            self.obs_hat_error_v_c = []
            self.obs_hat_error_all = []
            self.A11_a = []
            self.A11_b = []
            self.A11_c = []
            self.A12_a = []
            self.A12_b = []
            self.A12_c = []
            self.A21_a = []
            self.A21_b = []
            self.A21_c = []
            self.A22_a = []
            self.A22_b = []
            self.A22_c = []
            self.B1_a = []
            self.B1_b = []
            self.B1_c = []
            self.B2_a = []
            self.B2_b = []
            self.B2_c = []
            self.W_scale_list_a = []
            self.Poly_update_a = []
            self.W_scale_list_b = []
            self.Poly_update_b = []
            self.W_scale_list_c = []
            self.Poly_update_c = []
            self.poly_update_count = 0

        self.rew_sum = []

        [x.clear() for x in self.delay_queues]
        obs = super().reset()
        self.terminated = 0

        self._n_training_steps = 0
        self.used_P = np.zeros(self.action_space.shape)
        self.used_P_abc = np.zeros(self.action_space.shape)

        """
        Features
        """
        error = (obs[6:9] - obs[3:6]) / 2  # control error: v_setpoint - v_mess
        obs = np.append(obs, error)

        """
        Add used action to the NN input to learn delay
        """
        obs = np.append(obs, self.used_P)
        #obs = np.append(obs, self.used_P2)

        obs_delay_array = self.shift_and_append(obs[3:6])
        obs = np.append(obs, obs_delay_array)

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

            self.rewards.append(0)    # only for logging!!! Otherwise does alter the mean!!!
            self.rew_sum.append(0)

            self.safe_guard_failed.append(0)
            self.safe_guard_rand_failed.append(0)
            self.safe_guard_active.append(0)

            self.A_error.append(0)
            self.B_error.append(0)
            self.obs_hat_error.append(0)
            self.obs_hat_error_i_a.append(0)
            self.obs_hat_error_i_b.append(0)
            self.obs_hat_error_i_c.append(0)
            self.obs_hat_error_v_a.append(0)
            self.obs_hat_error_v_b.append(0)
            self.obs_hat_error_v_c.append(0)
            #self.obs_hat_error_all.append([0,0])
            self.A11_a.append(0)
            self.A11_b.append(0)
            self.A11_c.append(0)
            self.A12_a.append(0)
            self.A12_b.append(0)
            self.A12_c.append(0)
            self.A21_a.append(0)
            self.A21_b .append(0)
            self.A21_c.append(0)
            self.A22_a.append(0)
            self.A22_b.append(0)
            self.A22_c.append(0)
            self.B1_a.append(0)
            self.B1_b.append(0)
            self.B1_c.append(0)
            self.B2_a.append(0)
            self.B2_b.append(0)
            self.B2_c.append(0)
            self.W_scale_list_a.append(0)
            self.Poly_update_a.append(0)
            self.W_scale_list_b.append(0)
            self.Poly_update_b.append(0)
            self.W_scale_list_c.append(0)
            self.Poly_update_c.append(0)


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

    def init_model_fit(self):
        """ init constraint for safeguard, using initial model matrices """
        # calculate initial safeguard-constraints, based on rnd-system matrices
        n = 2  # self.env.observation_space.shape[0]
        m = 1  # self.env.action_space.shape[0]
        self.A_prev, self.B_prev = np.random.random((n, n)), np.random.random(
            (n, m))

        eigs = np.linalg.eigvals(self.A_prev)
        while not (np.abs(eigs) <= 1).all():
            # use only stable system matrice with eigvals within unit-circle
            self.A_prev, self.B_prev = np.random.random(
                (n, n)), np.random.random((n, m))
            eigs = np.linalg.eigvals(self.A_prev)

        #Mx, wx, Mu, wu, Q, R = self.state_space_mats
        feasible_init = calc_feasible_set(
            self.A_prev, self.B_prev, W_x, omega_x, W_u, omega_u,
            project_dim=[1, 2, 3],
            N_max=1,  # todo change for better polytope function, 1 iteration since matlab ok, but validate!!!
            tol=0.2,
            discount=0.9,
            N_start=1, # todo: ??
        )
        return feasible_init, self.A_prev, self.B_prev




    def update(
            self,
            model_estimate=None,
            estimate_variance=None,
            obs_hat_error=None,
            a=0
        ):
        """
        update safeguard model, model-fit parameter, ...

        args:

        """
        A_hat, B_hat = model_estimate

        if obs_hat_error > 0.01:
            W_scale = obs_hat_error * 20 + 1 # scales polytope to 1/10 * action if error = 2*obs_lim
        else:
            W_scale = 1
        if a == 0:
            self.W_scale_list_a.append(W_scale)
        if a == 1:
            self.W_scale_list_b.append(W_scale)
        if a == 2:
            self.W_scale_list_c.append(W_scale)

        if self.env.env.action_time_delay == 1:
            self.safe_guard.update_sys(A_hat, B_hat)
        #    self.safe_guard.A_d = A_hat
        #    self.safe_guard.B_d = B_hat

        #Mx, wx, Mu, wu, Q, R = self.state_space_mats
        var_y1, var_y2 = estimate_variance

        modulo_steps = self.total_steps % self.guard_update_steps
        #if (modulo_steps == 0) and (self.total_steps != 0):
        if True: #(modulo_steps == 0) or (self.total_steps < 1500):
            # difference to previous model-update
            delta_A = np.linalg.norm(
                A_hat - self.A_prev) / np.linalg.norm(
                self.A_prev)
            delta_B = np.linalg.norm(
                B_hat - self.B_prev) / np.linalg.norm(
                self.B_prev)
            #print(f'delta A: {delta_A},  steps: {self.total_steps}')

            # update safeguard constraint im change in model-est is significant
            if delta_A >= self.model_delta:
                self.poly_update_count += 1
                #print(f'\n\n  {delta_A} \nAold: {self.A_prev}, \nA: {A_hat}')
                if a == 0:
                    self.Poly_update_a.append(1)
                if a == 1:
                    self.Poly_update_b.append(1)
                if a == 2:
                    self.Poly_update_c.append(1)

                feasible_set = calc_feasible_set(
                    A_hat, B_hat, W_x*W_scale, omega_x, W_u*W_scale, omega_u,
                    project_dim=[1, 2, 3],
                    N_max=1,  # todo change for better polytope function, 1 iteration since matlab ok, but validate!!!
                    tol=0.2,
                    discount=0.8,
                    N_start=1,  #todo: ??
                    notebook_bar=False,
                    progress_bar=False,
                    verbose=0,
                    plt_all=False
                )

                self.safe_guard.update(
                    constraints=feasible_set,
                    fit_error_margin=np.array([var_y1, var_y2]),
                    mode='update',
                )

                self.A_prev, self.B_prev = A_hat, B_hat
                #self.model_fit_log['updated_episode'].append((episode))
            else:
                if a == 0:
                    self.Poly_update_a.append(0)
                if a == 1:
                    self.Poly_update_b.append(0)
                if a == 2:
                    self.Poly_update_c.append(0)
        else:
            if a == 0:
                self.Poly_update_a.append(0)
            if a == 1:
                self.Poly_update_b.append(0)
            if a == 2:
                self.Poly_update_c.append(0)


