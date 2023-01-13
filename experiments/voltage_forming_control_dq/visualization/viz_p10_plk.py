import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from openmodelica_microgrid_gym.util import dq0_to_abc
from feasible_set_test import feasible_set, A_d, B_d

from experiments.voltage_forming_control_dq.util.config import cfg

study_name = cfg['STUDY_NAME']
meas_data_folder = cfg['meas_data_folder']

# SL ohne Delay
trial = ['28']
episode_list = [ '0']#, '10', '52', '66', '75']
#episode_list = ['2', '34', '46' ,'60']
terminated_list = ['0', '0', '0', '0']
add_str = [ '_TEST', '_TEST', '_TEST']
RLS = 1
plt_R = 1

#study_name = 'P10_Safe_DDPG_RLS_DELAY_Poly_U_X_scaling_on_obsError_normalized_soft_constraints_shift_loadstep'
#trial = ['1']
#episode_list = ['50']
#terminated_list = ['0', '0']
#RLS = 1
#plt_R = 1


L = 70e-6
R1 = 1.1e-3
R2 = 7e-3

C = 250e-6

interval_list_x = [0, 0.11]
ts = 1e-4

v_dead = 650  # net -> v_lim
i_dead = 400  # net -> i_lim
V_dc = 400

for i in range(len(trial)):

    train_data = pd.read_pickle('data/'+ meas_data_folder + study_name +'/' + trial[i] +'/' + trial[i] + '_'+study_name+
                               '_training_rewards_trial_number_' + trial[i] +'.pkl.bz2')

    #train_data = pd.read_pickle('data/' + trial[i] + '_P10_SEC_R_load_traing_rewards_Trial_number_' + trial[i]

    #train_data = pd.read_pickle('data/experiment_data/' + study_name +'/' + trial[i] +'/' + trial[i] + '_'+study_name+
    #                            '_training_rewards_trial_number'
    #                           +'_'+ trial[i]+'.pkl.bz2')
    test_data = pd.read_pickle(
        'data/' + meas_data_folder + study_name + '/' + trial[i] + '/' + trial[i] + '_' + study_name +
        '_test_reward_trial_number_' + trial[i] + '.pkl.bz2')



    test_reward = test_data['Reward']
    t_reward = np.arange(0, round((len(test_reward)) * ts, 4), ts).tolist()

    plt.plot(t_reward, np.array(test_reward), 'b', label=f'      SEC-DDPG: '
                                            f'{round(sum(test_reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')


    plt.grid()
    plt.xlim(interval_list_x)
    plt.legend()
    plt.ylabel("Reward")
    plt.show()

    plt.plot(test_data['Reward'])
    plt.ylabel('Test Reward')
    plt.grid()
    plt.title(str(trial))
    plt.show()




    plt.plot(train_data['Mean_eps_env_reward_raw'])
    plt.ylabel('Train Reward raw (w/o penalty)')
    plt.grid()
    plt.title(str(trial))
    plt.show()

    plt.plot(train_data['Mean_eps_reward_sum'])
    plt.ylabel('Train Reward sum')
    plt.grid()
    plt.title(str(trial))
    plt.show()

    #fig, axs = plt.subplots(3, 1, figsize=(9, 7))
    plt.title(str(trial))
    plt.plot(train_data['Num_steps_per_episode'])
    plt.ylabel('num_Steps')
    plt.grid()
    plt.xlabel('Episodes')
    plt.show()

    cumsum_steps = train_data['Num_steps_per_episode'].cumsum()
    if RLS:
        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    plt.title(str(trial))
    axs[0].plot(train_data['Num_steps_per_episode'])
    axs[0].set_ylabel('num_Steps')
    #axs[0].set_xlim([22, 33])
    axs[0].grid()
    axs[1].plot(cumsum_steps)
    axs[1].set_ylabel('cumulated Steps')
    axs[1].grid()
    axs[1].set_ylim([0, 250])
    #axs[1].set_xlim([22, 33])

    if RLS:
        axs[2].plot(train_data['Poly_update_per_episode'])
        #axs[2].set_xlim([22, 33])
        axs[2].grid()
        axs[2].set_ylabel('num_poly_updates')

    plt.show()

    if RLS:
        plt.plot(train_data['A_error_mean'])
        plt.ylabel('Mean FrobeniusNorm A_hat')
        plt.grid()
        plt.xlabel('Epsiodes')
        plt.title(str(trial))
        plt.show()

        plt.plot(train_data['B_error_mean'])
        plt.ylabel('Mean FrobeniusNorm B_hat')
        plt.grid()
        plt.xlabel('Epsiodes')
        plt.title(str(trial))
        plt.show()
    
    ep_count = 0
    if episode_list is not None:
        for episode, terminated in zip(episode_list, terminated_list):
            episode_data = pd.read_pickle(
                'data/'+ meas_data_folder +  study_name + '/' + trial[i] + '/' + trial[i] + '_' + study_name +
                '_training_episode_number_'+ episode
                + '_terminated'+ terminated +add_str[ep_count]+'.pkl.bz2')
            ep_count += 1

            if plt_R:
                fig, axs = plt.subplots(5, 1, figsize=(9, 7))
            else:
                fig, axs = plt.subplots(4, 1, figsize=(9, 7))

            t = np.arange(0, round((len(episode_data['i_a_training'].to_list())) * ts, 4), ts).tolist()

            v_ref = dq0_to_abc(np.array([40, 0, 0]), episode_data['Phase'].to_list())


            axs[1].plot(t, episode_data['i_a_training'].to_list(), 'b')#, label='$\mathrm{DDPG}_\mathrm{}$')
            axs[1].plot(t, episode_data['i_b_training'].to_list(), 'r')
            axs[1].plot(t, episode_data['i_c_training'].to_list(), 'g')
            axs[1].grid()
            #axs[0, 0].legend()
            #axs[0].set_title(['Episode' + str(episode)])
            axs[1].set_xlim(interval_list_x)
            axs[1].set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{V}$')
            axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')
            #plt.show()

            axs[0].plot(t, episode_data['v_a_training'].to_list(), 'b')
            axs[0].plot(t, episode_data['v_b_training'].to_list(), 'r')
            axs[0].plot(t, episode_data['v_c_training'].to_list(), 'g')
            axs[0].plot(t, v_ref[0], ':', color='gray')
            axs[0].grid()
            #plt.legend()
            axs[0].set_title(['Episode' + str(episode)])
            axs[0].set_xlim(interval_list_x)
            axs[0].set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')

            axs[2].plot(t, episode_data['v_a_training'].to_list(), 'b')
            axs[2].plot(t, episode_data['v_b_training'].to_list(), 'r')
            axs[2].plot(t, episode_data['v_c_training'].to_list(), 'g')
            axs[2].grid()
            axs[2].set_xlim(interval_list_x)
            axs[2].set_ylabel('$u_{\mathrm{abc}}\,/\,\mathrm{V}$')

            #axs[3].plot(t, episode_data['Rewards_raw'].to_list(), 'b', label='$\mathrm{r}_\mathrm{env, unscaled}$')
            axs[3].plot(t, episode_data['Rewards_sum'].to_list(), 'r', label='$\mathrm{r}_\mathrm{punish, scaled}$')
            axs[3].grid()
            axs[3].set_xlim(interval_list_x)
            plt.legend(loc = "upper right")
            axs[3].set_ylabel('r')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[3].text(0.25, 0.95, "$\mathrm{r}_\mathrm{min}(-0.75) \cdot (1-\gamma) = -0.04$", transform=axs[3].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            if plt_R:
                axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                #axs[4].plot(t, episode_data['safe_guard_active'].to_list(), 'g')
                axs[4].set_ylabel('SG-active?')
                axs[4].grid()
                axs[4].set_xlim(interval_list_x)
                #axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                #axs[4].set_ylabel('$R_{\mathrm{Load}}\,/\,\mathrm{Ohm}$')

            plt.show()

            if RLS:
                R_load = episode_data['R_load_training'].to_numpy()
                A_sys = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L + R2 / (L * (R_load + R2))],
                                  [1 / (C * (1 + R2 / R_load)), -1 / (C * (R_load + R2))]])

                #B_sys = np.array([[1 / L], [0]])

                A_sys[0,1, :] = A_sys[0, 1, :] * v_dead / i_dead
                A_sys[1, 0, :] = A_sys[1, 0, :] * i_dead / v_dead
                #B_sys[0, 0] = B_sys[0, 0] * i_dead / V_dc

                A_d2 = np.ndarray(shape=(2,2,len(R_load)))

                for i in range(len(R_load)):
                    A_d2[:,:,i] = scipy.linalg.expm(A_sys[:,:,i] * ts)
                """
                fig, axs = plt.subplots(2, 1, figsize=(9, 7))
                axs[0].plot(t, episode_data['A_error'].to_list())
                axs[1].plot(t, episode_data['B_error'].to_list())
                axs[0].set_ylabel('FrobeniusNorm A_hat')
                axs[1].set_ylabel('FrobeniusNorm B_hat')
                axs[1].set_xlabel(r'$t\,/\,\mathrm{s}$')
                axs[0].grid()
                axs[1].grid()
                axs[0].set_title(['Episode' + str(episode)])
                plt.show()

                
                fig, axs = plt.subplots(2, 1, figsize=(9, 7))
                axs[0].plot(t, episode_data['obs_hat_error_i_a'].to_list(), 'b')
                axs[0].plot(t, episode_data['obs_hat_error_i_b'].to_list(), 'r')
                axs[0].plot(t, episode_data['obs_hat_error_i_c'].to_list(), 'g')
                axs[0].grid()
                axs[0].set_ylabel('obs_error_i_abc')
                axs[1].plot(t, episode_data['obs_hat_error_v_a'].to_list(), 'b')
                axs[1].plot(t, episode_data['obs_hat_error_v_b'].to_list(), 'r')
                axs[1].plot(t, episode_data['obs_hat_error_v_c'].to_list(), 'g')
                axs[1].grid()
                axs[1].set_ylabel('obs_error_v_abc')
                plt.show()
                """

                ones_vec = np.ones(len(t)-1)

                fig, axs = plt.subplots(2, 2, figsize=(9, 7))
                axs[0, 0].plot(t[1:], episode_data['A11_a'].to_list()[1:], 'b')
                #axs[0, 0].plot(t[1:], ones_vec*A_d[0, 0], ':', color='red')
                axs[0, 0].plot(t[1:], A_d2[0, 0, 1:], ':', color='red')
                axs[0, 0].grid()
                axs[0, 0].set_ylabel('Ad11_a')
                axs[0, 1].plot(t[1:], episode_data['A12_a'].to_list()[1:], 'b')
                #axs[0, 1].plot(t[1:], ones_vec*A_d[0, 1], ':', color='red')
                axs[0, 1].plot(t[1:], A_d2[0, 1, 1:], ':', color='red')
                axs[0, 1].grid()
                axs[0, 1].set_ylabel('Ad12_a')
                axs[1, 0].plot(t[1:], episode_data['A21_a'].to_list()[1:], 'b')
                #axs[1, 0].plot(t[1:], ones_vec*A_d[1, 0], ':', color='red')
                axs[1, 0].plot(t[1:], A_d2[1, 0, 1:], ':', color='red')
                axs[1, 0].grid()
                axs[1, 0].set_ylabel('Ad21_a')
                axs[1, 1].plot(t[1:], episode_data['A22_a'].to_list()[1:], 'b')
                #axs[1, 1].plot(t[1:], ones_vec*A_d[1, 1], ':', color='red')
                axs[1, 1].plot(t[1:], A_d2[1, 1, 1:], ':', color='red')
                axs[1, 1].grid()
                axs[1, 1].set_ylabel('Ad22_a')
                plt.show()
                time.sleep(0.5)
                """
                fig, axs = plt.subplots(2, 2, figsize=(9, 7))
                axs[0, 0].plot(t[1:], episode_data['A11_b'].to_list()[1:], 'b')
                axs[0, 0].plot(t[1:], ones_vec*A_d[0, 0], ':', color='red')
                axs[0, 0].grid()
                axs[0, 0].set_ylabel('Ad11_b')
                axs[0, 1].plot(t[1:], episode_data['A12_b'].to_list()[1:], 'b')
                axs[0, 1].plot(t[1:], ones_vec*A_d[0, 1], ':', color='red')
                axs[0, 1].grid()
                axs[0, 1].set_ylabel('Ad12_b')
                axs[1, 0].plot(t[1:], episode_data['A21_b'].to_list()[1:], 'b')
                axs[1, 0].plot(t[1:], ones_vec*A_d[1, 0], ':', color='red')
                axs[1, 0].grid()
                axs[1, 0].set_ylabel('Ad21_b')
                axs[1, 1].plot(t[1:], episode_data['A22_b'].to_list()[1:], 'b')
                axs[1, 1].plot(t[1:], ones_vec*A_d[1, 1], ':', color='red')
                axs[1, 1].grid()
                axs[1, 1].set_ylabel('Ad22_b ')
                plt.show()
                time.sleep(0.5)

                fig, axs = plt.subplots(2, 2, figsize=(9, 7))
                axs[0, 0].plot(t[1:], episode_data['A11_c'].to_list()[1:], 'b')
                axs[0, 0].plot(t[1:], ones_vec * A_d[0, 0], ':', color='red')
                axs[0, 0].grid()
                axs[0, 0].set_ylabel('Ad11_c')
                axs[0, 1].plot(t[1:], episode_data['A12_c'].to_list()[1:], 'b')
                axs[0, 1].plot(t[1:], ones_vec * A_d[0, 1], ':', color='red')
                axs[0, 1].grid()
                axs[0, 1].set_ylabel('Ad12_c')
                axs[1, 0].plot(t[1:], episode_data['A21_c'].to_list()[1:], 'b')
                axs[1, 0].plot(t[1:], ones_vec * A_d[1, 0], ':', color='red')
                axs[1, 0].grid()
                axs[1, 0].set_ylabel('Ad21_c')
                axs[1, 1].plot(t[1:], episode_data['A22_c'].to_list()[1:], 'b')
                axs[1, 1].plot(t[1:], ones_vec * A_d[1, 1], ':', color='red')
                axs[1, 1].grid()
                axs[1, 1].set_ylabel('Ad22_c ')
                plt.show()
                time.sleep(0.5)

                fig, axs = plt.subplots(2, 1, figsize=(9, 7))
                axs[0].plot(t[1:], episode_data['B1_a'].to_list()[1:], 'b')
                axs[0].plot(t[1:], ones_vec * B_d[0, 0], ':', color='red')
                axs[0].grid()
                axs[0].set_ylabel('B1_a')
                axs[1].plot(t[1:], episode_data['B2_a'].to_list()[1:], 'b')
                axs[1].plot(t[1:], ones_vec * B_d[1, 0], ':', color='red')
                axs[1].grid()
                axs[1].set_ylabel('B2_a')
                plt.show()
                time.sleep(0.5)

                fig, axs = plt.subplots(2, 1, figsize=(9, 7))
                axs[0].plot(t[1:], episode_data['B1_b'].to_list()[1:], 'b')
                axs[0].plot(t[1:], ones_vec * B_d[0, 0], ':', color='red')
                axs[0].grid()
                axs[0].set_ylabel('B1_b')
                axs[1].plot(t[1:], episode_data['B2_b'].to_list()[1:], 'b')
                axs[1].plot(t[1:], ones_vec * B_d[1, 0], ':', color='red')
                axs[1].grid()
                axs[1].set_ylabel('B2_b')
                plt.show()
                time.sleep(0.5)

                fig, axs = plt.subplots(2, 1, figsize=(9, 7))
                axs[0].plot(t[1:], episode_data['B1_c'].to_list()[1:], 'b')
                axs[0].plot(t[1:], ones_vec * B_d[0, 0], ':', color='red')
                axs[0].grid()
                axs[0].set_ylabel('B1_c')
                axs[1].plot(t[1:], episode_data['B2_c'].to_list()[1:], 'b')
                axs[1].plot(t[1:], ones_vec * B_d[1, 0], ':', color='red')
                axs[1].grid()
                axs[1].set_ylabel('B2_c')
                plt.show()
                time.sleep(0.5)
                """






        asd = 1


