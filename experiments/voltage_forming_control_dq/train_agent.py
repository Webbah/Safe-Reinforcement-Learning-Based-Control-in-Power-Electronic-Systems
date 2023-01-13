import platform
from os import makedirs

import gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# np.random.seed(0)

from experiments.voltage_forming_control_dq.envs.env_wrapper import BaseWrapper
from experiments.voltage_forming_control_dq.envs.rewards import Reward
from experiments.voltage_forming_control_dq.envs.sec_wrapper import SecWrapperPastVals, SecWrapper
from experiments.voltage_forming_control_dq.envs.vctrl_single_inv import net
from experiments.voltage_forming_control_dq.util.config import cfg
from experiments.voltage_forming_control_dq.util.custom_network import custom_network
from experiments.voltage_forming_control_dq.util.td3_custom_policy import CustomTD3Policy
# from experiments.voltage_forming_control_dq.util.td3_custom_policy import register_td3_custom_policy

folder_name = cfg['STUDY_NAME']
node = platform.uname().node


def train_ddpg(HPO, learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale, alpha_relu_actor,
               batch_size,
               actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
               alpha_relu_critic,
               noise_var, noise_theta, error_exponent,
               training_episode_length, buffer_size,  # learning_starts,
               tau, number_learning_steps, integrator_weight, antiwindup_weight,
               penalty_I_weight, penalty_P_weight,
               train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
               number_past_vals, seed, n_trial, safe_layer, learn_model):
    np.random.seed(seed)
    if node in cfg['lea_vpn_nodes']:
        save_folder = 'data/' + cfg['meas_data_folder']
        if node in ['lea-cyberdyne']:
            save_folder = cfg['meas_data_folder']
        log_path = f'{folder_name}/{n_trial}/'
    else:
        # assume we are on a node of pc2 -
        save_folder = cfg['pc2_logpath'] + '/' + cfg['meas_data_folder']
        pc2_log_path = cfg['pc2_logpath']
        log_path = f'{pc2_log_path}/{folder_name}/{n_trial}/'

    makedirs(save_folder, exist_ok=True)

    rew = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, gamma,
                 use_gamma_normalization=use_gamma_in_rew, error_exponent=error_exponent, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

    env = gym.make('experiments.voltage_forming_control_dq.envs:vctrl_single_inv_train-v0',
                   reward_fun=rew.rew_fun_dq0,
                   abort_reward=-1,
                   obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                               'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                               'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2']
                   )

    if cfg['env_wrapper'] == 'past':
        env = SecWrapperPastVals(env, number_of_features=9 + number_past_vals * 3,
                                 training_episode_length=training_episode_length,
                                 # recorder=mongo_recorder,
                                 n_trail=n_trial, integrator_weight=integrator_weight,
                                 antiwindup_weight=antiwindup_weight, gamma=gamma,
                                 penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                                 t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                                 number_learing_steps=number_learning_steps, number_past_vals=number_past_vals,
                                 config=cfg, safe=safe_layer)

    elif cfg['env_wrapper'] == 'no-I-term':
        env = BaseWrapper(env, number_of_features=6 + number_past_vals * 3,
                          training_episode_length=training_episode_length,
                          # recorder=mongo_recorder,
                          n_trail=n_trial, gamma=gamma,
                          number_learing_steps=number_learning_steps, number_past_vals=number_past_vals, config=cfg,
                          safe=safe_layer, learn_model=learn_model)

    else:
        env = SecWrapper(env, number_of_features=11, training_episode_length=training_episode_length,
                         # recorder=mongo_recorder,
                         n_trail=n_trial, integrator_weight=integrator_weight,
                         antiwindup_weight=antiwindup_weight, gamma=gamma,
                         penalty_I_weight=penalty_I_weight, penalty_P_weight=penalty_P_weight,
                         t_start_penalty_I=t_start_penalty_I, t_start_penalty_P=t_start_penalty_P,
                         number_learing_steps=number_learning_steps)  # , use_past_vals=True, number_past_vals=30)

    # increase action space to generate model with 6 outputs if SEC
    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env.action_space = gym.spaces.Box(low=np.full(6, -1), high=np.full(6, 1))

    n_actions = env.action_space.shape[-1]
    noise_var = noise_var
    noise_theta = noise_theta  # stiffness of OU
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=noise_theta * np.ones(n_actions),
                                                sigma=noise_var * np.ones(n_actions), dt=net.ts)

    print(optimizer)
    if optimizer == 'SGD':
        used_optimzer = th.optim.SGD
    elif optimizer == 'RMSprop':
        used_optimzer = th.optim.RMSprop
    else:
        used_optimzer = th.optim.Adam

    # policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
    #                                                                  ,qf=[critic_hidden_size] * critic_number_layers),
    #                     optimizer_class=used_optimzer)

    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
                                                                      , qf=[critic_hidden_size] * critic_number_layers),
                         optimizer_class=used_optimzer,
                         alpha_relu_actor=alpha_relu_actor,
                         alpha_relu_critic=alpha_relu_critic,
                         weight_scale = weight_scale, bias_scale = bias_scale)

    # model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=log_path,
    model = DDPG('CustomTD3Policy', env, verbose=1, tensorboard_log=log_path,
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=buffer_size,
                 # learning_starts=int(learning_starts * training_episode_length),
                 batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                 train_freq=(train_freq, train_freq_type), gradient_steps=- 1,
                 optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)

    # is shifted to CustomTD3Policy based on the readme of sb3
    """
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=dict(pi=[actor_hidden_size] * actor_number_layers
                                                                      , qf=[critic_hidden_size] * critic_number_layers),
                         optimizer_class=used_optimzer)
    model2 = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=log_path,
                 policy_kwargs=policy_kwargs,
                 learning_rate=learning_rate, buffer_size=buffer_size,
                 # learning_starts=int(learning_starts * training_episode_length),
                 batch_size=batch_size, tau=tau, gamma=gamma, action_noise=action_noise,
                 train_freq=(train_freq, train_freq_type), gradient_steps=- 1,
                 optimize_memory_usage=False,
                 create_eval_env=False, seed=None, device='auto', _init_setup_model=True)
    model2 = custom_network(model2, actor_number_layers, critic_number_layers,
                            weight_scale, bias_scale, alpha_relu_actor, alpha_relu_critic)
""  """
    if cfg['env_wrapper'] not in ['no-I-term', 'I-controller']:
        env.action_space = gym.spaces.Box(low=np.full(3, -1), high=np.full(3, 1))

    # start training
    model.learn(total_timesteps=number_learning_steps)

    if env.model_id is not None:
        # Log Train-info data
        train_data = {
            "Mean_eps_env_reward_raw": env.reward_episode_mean,
            "Mean_eps_reward_sum": env.reward_plus_addon_episode_mean,
            "Num_steps_per_episode": env.steps_per_episode,
            "Terminated_in_epsidode": env.terminated,
            "A_error_mean": env.A_error_mean_per_episode,
            "B_error_mean": env.B_error_mean_per_episode,
            "obs_hat_error_mea_per_episode": env.obs_hat_error_mea_per_episode,
            "Poly_update_per_episode": env.Poly_update_per_episode,
        }
    else:
        train_data = {
            "Mean_eps_env_reward_raw": env.reward_episode_mean,
            "Mean_eps_reward_sum": env.reward_plus_addon_episode_mean,
            "Num_steps_per_episode": env.steps_per_episode,
            "Terminated_in_epsidode": env.terminated,
        }

    df = pd.DataFrame(train_data)
    makedirs(save_folder + log_path, exist_ok=True)
    df.to_pickle(save_folder + log_path + str(n_trial) + '_' + folder_name + '_' +
                 'training_rewards_trial_number_' + n_trial + ".pkl.bz2")

    model.save(log_path + f'model.zip')
    if HPO:
        ####### Run Test #########
        return_sum = 0.0
        rew.gamma = 0
        # episodes will not abort, if limit is exceeded reward = -1
        rew.det_run = True
        env.deterministic_run = True
        rew.exponent = 0.5  # 1
        limit_exceeded_in_test = False


        if safe_layer == 1:
            safe_layer = 2
        # todo
        print("Test and training executed on 100 ohm!!!")
        env_test = gym.make('experiments.voltage_forming_control_dq.envs:vctrl_single_inv_test-v0',
        #env_test = gym.make('experiments.voltage_forming_control_dq.envs:vctrl_single_inv_train-v0',
                            reward_fun=rew.rew_fun_dq0,
                            abort_reward=-1,  # no needed if in rew no None is given back
                            # on_episode_reset_callback=cb.fire  # needed?
                            obs_output=['lc.inductor1.i', 'lc.inductor2.i', 'lc.inductor3.i',
                                        'lc.capacitor1.v', 'lc.capacitor2.v', 'lc.capacitor3.v',
                                        'inverter1.v_ref.0', 'inverter1.v_ref.1', 'inverter1.v_ref.2'],
                            #max_episode_steps=1000
                            )

        if cfg['env_wrapper'] == 'past':
            env_test = SecWrapperPastVals(env_test, number_of_features=9 + number_past_vals * 3,
                                          integrator_weight=integrator_weight,
                                          # recorder=mongo_recorder,
                                          antiwindup_weight=antiwindup_weight,
                                          gamma=1, penalty_I_weight=0,
                                          penalty_P_weight=0, number_past_vals=number_past_vals,
                                          training_episode_length=training_episode_length, config=cfg, safe=safe_layer)

        elif cfg['env_wrapper'] == 'no-I-term':
            env_test = BaseWrapper(env_test, number_of_features=6 + number_past_vals * 3,
                                   training_episode_length=training_episode_length,
                                   # recorder=mongo_recorder,
                                   n_trail=n_trial, gamma=gamma,
                                   number_learing_steps=number_learning_steps, number_past_vals=number_past_vals,
                                   config=cfg,
                                   safe=safe_layer, SG=env.safe_guard, RLS_model=env.model_id, learn_model=learn_model)

        else:
            env_test = SecWrapper(env_test, number_of_features=11, integrator_weight=integrator_weight,
                                  # recorder=mongo_recorder,
                                  antiwindup_weight=antiwindup_weight,
                                  gamma=1, penalty_I_weight=0,
                                  penalty_P_weight=0,
                                  training_episode_length=training_episode_length)  # , use_past_vals=True, number_past_vals=30)
        # using gamma=1 and rew_weigth=3 we get the original reward from the env without penalties
        obs = env_test.reset()
        phase_list = []
        phase_list.append(env_test.env.net.components[0].phase)
        env_test.log_name_added = "_TEST"
        rew_list = []
        env_test.max_episode_steps = 10000
        env_test.training_episode_length = 10000

        for step in range(env_test.max_episode_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env_test.step(action)
            if cfg['loglevel'] in ['train', 'test']:
                phase_list.append(env_test.env.net.components[0].phase)

            if rewards == -1 and not limit_exceeded_in_test:  # and env_test.rew[-1]:
                # Set addidional penalty of -1 if limit is exceeded once in the test case
                limit_exceeded_in_test = True

            if limit_exceeded_in_test:
                # if limit was exceeded once, reward will be kept to -1 till the end of the episode,
                # nevertheless what the agent does
                rewards = -1

            env_test.render()
            return_sum += rewards
            rew_list.append(rewards)

            # if step % 1000 == 0 and step != 0:
            #    env.close()
            #    obs = env.reset()

            if done:
                env_test.close()
                break

        reward_test_after_training = {"Reward": rew_list}

        df_test = pd.DataFrame(reward_test_after_training)
        df_test.to_pickle(save_folder + log_path + str(n_trial) + '_' + folder_name + '_' +
                          'test_reward_trial_number_' + n_trial + ".pkl.bz2")

        if env_test.safe_guard is not None:
            if env_test.model_id is not None:
                episode_data = {
                    "R_load_training": env_test.R_training,
                    "i_a_training": env_test.i_a,
                    "i_b_training": env_test.i_b,
                    "i_c_training": env_test.i_c,
                    "v_a_training": env_test.v_a,
                    "v_b_training": env_test.v_b,
                    "v_c_training": env_test.v_c,
                    "u_a_RL": env_test.u_a_RL,
                    "u_b_RL": env_test.u_b_RL,
                    "u_c_RL": env_test.u_c_RL,
                    "u_a_safe": env_test.u_a_safe,
                    "u_b_safe": env_test.u_b_safe,
                    "u_c_safe": env_test.u_c_safe,
                    # "slack": self.slack,
                    "A_error": env_test.A_error,
                    "B_error": env_test.B_error,
                    "obs_hat_error": env_test.obs_hat_error,
                    "obs_hat_error_i_a": env_test.obs_hat_error_i_a,
                    "obs_hat_error_i_b": env_test.obs_hat_error_i_b,
                    "obs_hat_error_i_c": env_test.obs_hat_error_i_c,
                    "obs_hat_error_v_a": env_test.obs_hat_error_v_a,
                    "obs_hat_error_v_b": env_test.obs_hat_error_v_b,
                    "obs_hat_error_v_c": env_test.obs_hat_error_v_c,
                    "Rewards_raw": env_test.rewards,
                    "Rewards_sum": env_test.rew_sum,
                    "Phase": env_test.phase,
                    "A11_a": env_test.A11_a,
                    "A11_b": env_test.A11_b,
                    "A11_c": env_test.A11_c,
                    "A12_a": env_test.A12_a,
                    "A12_b": env_test.A12_b,
                    "A12_c": env_test.A12_c,
                    "A21_a": env_test.A21_a,
                    "A21_b": env_test.A21_b,
                    "A21_c": env_test.A21_c,
                    "A22_a": env_test.A22_a,
                    "A22_b": env_test.A22_b,
                    "A22_c": env_test.A22_c,
                    "B1_a": env_test.B1_a,
                    "B1_b": env_test.B1_b,
                    "B1_c": env_test.B1_c,
                    "B2_a": env_test.B2_a,
                    "B2_b": env_test.B2_b,
                    "B2_c": env_test.B2_c,
                    "W_scale_a": env_test.W_scale_list_a,
                    "Poly_update_a": env_test.Poly_update_a,
                    "W_scale_b": env_test.W_scale_list_b,
                    "Poly_update_b": env_test.Poly_update_b,
                    "W_scale_c": env_test.W_scale_list_c,
                    "Poly_update_c": env_test.Poly_update_c,
                    "Solution_not_found": env_test.safe_guard_failed,
                    "Rand_solution_not_found": env_test.safe_guard_rand_failed
                }
                poly_vertices = {
                    "scaled_vertices": env_test.safe_guard.vertices
                }
            else:
                episode_data = {
                    "R_load_training": env_test.R_training,
                    "i_a_training": env_test.i_a,
                    "i_b_training": env_test.i_b,
                    "i_c_training": env_test.i_c,
                    "v_a_training": env_test.v_a,
                    "v_b_training": env_test.v_b,
                    "v_c_training": env_test.v_c,
                    "u_a_RL": env_test.u_a_RL,
                    "u_b_RL": env_test.u_b_RL,
                    "u_c_RL": env_test.u_c_RL,
                    "u_a_safe": env_test.u_a_safe,
                    "u_b_safe": env_test.u_b_safe,
                    "u_c_safe": env_test.u_c_safe,
                    # "slack": self.slack,
                    "Rewards_raw": env_test.rewards,
                    "Rewards_sum": env_test.rew_sum,
                    "Phase": env_test.phase,
                    "Solution_not_found": env_test.safe_guard_failed,
                    "Rand_solution_not_found": env_test.safe_guard_rand_failed,
                    "safe_guard_active": env_test.safe_guard_active
                }
        else:
            episode_data = {
                "R_load_training": env_test.R_training,
                "i_a_training": env_test.i_a,
                "i_b_training": env_test.i_b,
                "i_c_training": env_test.i_c,
                "v_a_training": env_test.v_a,
                "v_b_training": env_test.v_b,
                "v_c_training": env_test.v_c,
                "u_a_RL": env_test.u_a_RL,
                "u_b_RL": env_test.u_b_RL,
                "u_c_RL": env_test.u_c_RL,
                "Rewards_raw": env_test.rewards,
                "Rewards_sum": env_test.rew_sum,
                "Phase": env_test.phase,
            }

        n_trail = str(env_test.n_trail)

        save_folder = 'data/' + cfg['meas_data_folder']
        log_path = f'{cfg["STUDY_NAME"]}/{n_trail}/'

        df = pd.DataFrame(episode_data)
        makedirs(save_folder + log_path, exist_ok=True)
        df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                     'training_episode_number_' + str(env_test.n_episode) + '_terminated' + str(env_test.terminated) +
                     env_test.log_name_added + ".pkl.bz2")

        df = pd.DataFrame(env_test.safe_guard.vertices)
        makedirs(save_folder + log_path, exist_ok=True)
        df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                     'vertices_episode_number_' + str(env_test.n_episode) + '_terminated' + str(env_test.terminated) +
                     env_test.log_name_added + ".pkl.bz2")
        """
        slack_data = {"slack": env.safe_guard.slack_list}
        df = pd.DataFrame(slack_data)
        makedirs(save_folder + log_path, exist_ok=True)
        df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                     'slack_episode_number_' + str(env.n_episode) + '_terminated' + str(env.terminated) +
                     env.log_name_added + ".pkl.bz2")

        df = pd.DataFrame(env.safe_guard.vertices_slack)
        makedirs(save_folder + log_path, exist_ok=True)
        df.to_pickle(save_folder + log_path + n_trail + '_' + cfg["STUDY_NAME"] + '_' +
                     'vertices_slack_episode_number_' + str(env.n_episode) + '_terminated' + str(
            env.terminated) +
                     "env.log_name_added +.pkl.bz2")
        """

        return return_sum / 10000  # env.max_episode_steps
    else:
        return model
