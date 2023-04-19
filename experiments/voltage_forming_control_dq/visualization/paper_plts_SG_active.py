import time
from os import makedirs

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from openmodelica_microgrid_gym.util import dq0_to_abc, abc_to_dq0
from feasible_set_test import feasible_set, A_d, B_d, v_dead, i_dead

from experiments.voltage_forming_control_dq.util.config import cfg






study_name = '370_vDC_SL_RLS_Rconst_2'  # PAPER Fig. 5
trial = ['4']
episode_list = ['16']
meas_data_folder = cfg['meas_data_folder']
#episode_list = ['16']
terminated_list = ['0']
add_str = ['','_TEST']
RLS = 1
plt_R = 0


interval_list_x = [0.064, 0.075]
#interval_list_x = [-0.001, 0.28]
#interval_list_x = [-0.001, 0.11]
#interval_list_x = [0.0, 0.001]
xlim = True
ylim = False

save_results = True
folder_name = "plots_paper"
save_name = 'training_timeseries_370_vDC_SL_RLS_varLoad_25'
makedirs(folder_name, exist_ok=True)


L = 70e-6
R1 = 1.1e-3
R2 = 7e-3

C = 250e-6

ts = 1e-4

for tr in range(len(trial)):


    ep_count = 0
    if episode_list is not None:
        for episode, terminated in zip(episode_list, terminated_list):
            episode_data = pd.read_pickle(
                'data/'+ meas_data_folder +  study_name + '/' + trial[tr] + '/' + trial[tr] + '_' + study_name +
                #'data/' + meas_data_folder + '/' + trial[tr] + '_' + study_name +
                '_training_episode_number_' + episode
                + '_terminated' + terminated + add_str[ep_count] + '.pkl.bz2')
                #+ '.pkl.bz2')
            ep_count += 1

            if save_results:
                params = {'backend': 'ps',
                          'text.latex.preamble': [r'\usepackage{gensymb}'
                                                  r'\usepackage{amsmath,amssymb,mathtools}'
                                                  r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                                  r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
                          'axes.labelsize': 12.5,  # fontsize for x and y labels (was 10)
                          'axes.titlesize': 12.5,
                          'font.size': 12.5,  # was 10
                          'legend.fontsize': 12.5,  # was 10
                          'xtick.labelsize': 12,
                          'ytick.labelsize': 12,
                          'text.usetex': True,
                          'figure.figsize': [6.5, 5.5],  # [4.5, 7.5],
                          'font.family': 'serif',
                          'lines.linewidth': 1.2
                          }
                matplotlib.rcParams.update(params)

            if plt_R:
                fig, axs = plt.subplots(5, 1)#, figsize=(9, 7))
            else:
                fig, axs = plt.subplots(5, 1)#, figsize=(9, 7))

            t = np.arange(0, round((len(episode_data['i_a_training'].to_list())) * ts, 4), ts).tolist()
            t_action = np.arange(0, round((len(episode_data['i_a_training'].to_list())-1) * ts, 4), ts).tolist()
            # since u ist zero padded by one 0 in the beginning for logging we have to delete that 0 here

            v_ref = dq0_to_abc(np.array([325.27, 0, 0]), episode_data['Phase'].to_list())

            axs[1].plot(t, episode_data['i_a_training'].to_list(), 'c')  # , label='$\mathrm{DDPG}_\mathrm{}$')
            axs[1].plot(t, episode_data['i_b_training'].to_list(), 'm')
            axs[1].plot(t, episode_data['i_c_training'].to_list(), 'y')
            axs[1].grid()
            axs[1].set_xticklabels([])
            #axs[1].set_ylim([-60, 60])
            axs[1].tick_params(direction='in')
            # axs[0, 0].legend()
            # axs[0].set_title(['Episode' + str(episode)])
            if xlim:
                axs[1].set_xlim(interval_list_x)
            axs[1].set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{A}$')
            axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')
            # plt.show()

            axs[0].plot(t, episode_data['v_a_training'].to_list(), 'c')
            axs[0].plot(t, episode_data['v_b_training'].to_list(), 'm')
            axs[0].plot(t, episode_data['v_c_training'].to_list(), 'y')
            axs[0].plot(t, v_ref[0], ':', color='gray')
            axs[0].grid()
            axs[0].set_xticklabels([])
            axs[0].tick_params(direction='in')
            # plt.legend()
            #axs[0].set_title(['Episode' + str(episode)])
            if xlim:
                axs[0].set_xlim(interval_list_x)
            axs[0].set_ylabel('$v_{\mathrm{abc}}\,/\,\mathrm{V}$')


            axs[2].plot(t_action, (episode_data['u_a_safe']).to_list()[1:], 'c')
            axs[2].plot(t_action, (episode_data['u_b_safe']).to_list()[1:], 'm')
            axs[2].plot(t_action, (episode_data['u_c_safe']).to_list()[1:], 'y')

            #axs[2].plot(t_action, (episode_data['u_a_RL']).to_list()[1:], 'c')
            #axs[2].plot(t_action, (episode_data['u_b_RL']).to_list()[1:], 'm')
            #axs[2].plot(t_action, (episode_data['u_c_RL']).to_list()[1:], 'y')
            axs[2].grid()
            axs[2].scatter(0.0675, 0, s=100, facecolors='none', edgecolors='green', linewidth=2)
            axs[2].set_xticklabels([])
            axs[2].tick_params(direction='in')
            if ylim:
                axs[2].set_ylim([-2.5, 0.5])
            if xlim:
                axs[2].set_xlim(interval_list_x)
            axs[2].set_ylabel('$\\tilde{u}_{\mathrm{abc,SG}}$')
            #axs[2].set_ylabel('$u_{\mathrm{abc,RL}}$')

            """
            axs[3].plot(t, (episode_data['u_a_RL'] * 370).to_list(), 'b')
            axs[3].plot(t, (episode_data['u_b_RL'] * 370).to_list(), 'r')
            axs[3].plot(t, (episode_data['u_c_RL'] * 370).to_list(), 'g')
            """

            axs[3].plot(t_action, (episode_data['u_a_RL'] ).to_list()[1:], 'c')
            axs[3].plot(t_action, (episode_data['u_b_RL'] ).to_list()[1:], 'm')
            axs[3].plot(t_action, (episode_data['u_c_RL'] ).to_list()[1:], 'y')
            axs[3].scatter(0.0675, -0.98, s=100, facecolors='none', edgecolors='green', linewidth=2)
            axs[3].grid()
            axs[3].set_xticklabels([])
            axs[3].tick_params(direction='in')
            #axs[3].set_ylim([-920, 200])
            if ylim:
                axs[3].set_ylim([-2.5, 0.5])
            #axs[3].set_ylim([-0.2, 0.2])
            if xlim:
                axs[3].set_xlim(interval_list_x)
            axs[3].set_ylabel('$\\tilde{u}_{\mathrm{abc,RL}}$')


            #axs[3].plot(t, episode_data['Rewards_raw'].to_list(), 'b', label='$\mathrm{r}_\mathrm{env, unscaled}$')
            axs[4].plot(t_action, episode_data['Rewards_sum'].to_list()[1:], 'k', label='$\mathrm{r}_\mathrm{punish, scaled}$')
            axs[4].grid()
            # plt.legend(loc="upper right")
            axs[4].set_ylabel('$r$')
            axs[4].tick_params(direction='in')
            if xlim:
                axs[4].set_xlim(interval_list_x)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            #axs[4].text(0.25, 0.95, "$\mathrm{r}_\mathrm{min}(-0.75) \cdot (1-\gamma) = -0.04$",
             #           transform=axs[3].transAxes, fontsize=14,
             #           verticalalignment='top', bbox=props)

            if plt_R:
                axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                # axs[4].plot(t, episode_data['safe_guard_active'].to_list(), 'g')
                #axs[4].set_ylabel('SG-active?')
                axs[4].grid()
                if xlim:
                    axs[4].set_xlim(interval_list_x)
                # axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                axs[4].set_ylabel('$R_{\mathrm{Load}}\,/\,\mathrm{\Omega}$')
                axs[4].set_xlabel('$t_{\mathrm{}}\,/\,\mathrm{s}$')
                axs[3].set_xticklabels([])
                axs[4].tick_params(direction='in')
            else:
                axs[4].set_xlabel('$t_{\mathrm{}}\,/\,\mathrm{s}$')
            fig.align_ylabels()
            plt.show()
            time.sleep(0.5)

            if save_results:
                fig.savefig(f'{folder_name}/' + study_name + 'timeseries.png', bbox_inches='tight', dpi=500)
                fig.savefig(f'{folder_name}/' + study_name + 'timeseries.pgf', bbox_inches='tight')
                fig.savefig(f'{folder_name}/' + study_name + 'timeseries.pdf', bbox_inches='tight')
                #fig.savefig(f'{folder_name}/' + save_name + 'timeseries_testcase.png', bbox_inches='tight', dpi=500)
                #fig.savefig(f'{folder_name}/' + save_name + 'timeseries_testcase.pgf', bbox_inches='tight')
                #fig.savefig(f'{folder_name}/' + save_name + 'timeseries_testcase.pdf', bbox_inches='tight')

            phase = episode_data['Phase'].tolist()  # env_test.env.net.components[0].phase
            v_dq0 = abc_to_dq0(np.array([episode_data['v_a_training'], episode_data['v_b_training'], episode_data['v_c_training']]), phase)
            i_dq0 = abc_to_dq0(np.array([episode_data['i_a_training'], episode_data['i_b_training'], episode_data['i_c_training']]), phase)
            u_dq0 = abc_to_dq0(np.array([episode_data['u_a_safe'], episode_data['u_b_safe'], episode_data['u_c_safe']]), phase)
            u_dq0_RL = abc_to_dq0(np.array([episode_data['u_a_RL'], episode_data['u_b_RL'], episode_data['u_c_RL']]), phase)

            if plt_R:
                fig, axs = plt.subplots(5, 1, figsize=(9, 7))
            else:
                fig, axs = plt.subplots(4, 1, figsize=(9, 7))

            t = np.arange(0, round((len(episode_data['i_a_training'].to_list())) * ts, 4), ts).tolist()
            v_d_ref = [325.27] * len(t)
            v_d_ref0 = [0] * len(t)

            axs[1].plot(t, i_dq0[0].tolist(), 'b')  # , label='$\mathrm{DDPG}_\mathrm{}$')
            axs[1].plot(t, i_dq0[1].tolist(), 'r')
            axs[1].plot(t, i_dq0[2].tolist(), 'g')
            axs[1].grid()
            # axs[0, 0].legend()
            # axs[0].set_title(['Episode' + str(episode)])
            if xlim:
                axs[1].set_xlim(interval_list_x)
            axs[1].set_ylabel('$i_{\mathrm{abc}}\,/\,\mathrm{V}$')
            axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')
            # plt.show()

            axs[0].plot(t, v_dq0[0].tolist(), 'b')
            axs[0].plot(t, v_dq0[1].tolist(), 'r')
            axs[0].plot(t, v_dq0[2].tolist(), 'g')
            axs[0].plot(t, v_d_ref, ':', color='gray')
            #axs[0].plot(t, v_d_ref0, ':', color='gray')
            axs[0].grid()
            #axs[0].set_ylim([280, 380])
            # plt.legend()
            axs[0].set_title(['Episode' + str(episode)])
            if xlim:
                axs[0].set_xlim(interval_list_x)
            axs[0].set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')

            axs[2].plot(t, (u_dq0[0]*370), 'b')
            axs[2].plot(t, (u_dq0[1]*370), 'r')
            axs[2].plot(t, (u_dq0[2]*370), 'g')
            axs[2].grid()
            if xlim:
                axs[2].set_xlim(interval_list_x)
            axs[2].set_ylabel('$v_{\mathrm{i,abc}}\,/\,\mathrm{V}$')

            #axs[3].plot(t, episode_data['Rewards_raw'].to_list(), 'b', label='$\mathrm{r}_\mathrm{env, unscaled}$')
            axs[3].plot(t, episode_data['Rewards_sum'].to_list(), 'r', label='$\mathrm{r}_\mathrm{punish, scaled}$')
            axs[3].grid()
            plt.legend(loc="upper right")
            axs[3].set_ylabel('r')
            if xlim:
                axs[3].set_xlim(interval_list_x)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[3].text(0.25, 0.95, "$\mathrm{r}_\mathrm{min}(-0.75) \cdot (1-\gamma) = -0.04$",
                        transform=axs[3].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

            if plt_R:
                axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                # axs[4].plot(t, episode_data['safe_guard_active'].to_list(), 'g')
                #axs[4].set_ylabel('SG-active?')
                axs[4].grid()
                if xlim:
                    axs[4].set_xlim(interval_list_x)
                # axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
                axs[4].set_ylabel('$R_{\mathrm{Load}}\,/\,\mathrm{Ohm}$')

            plt.show()
            time.sleep(0.5)




            fig, axs = plt.subplots(4, 1, figsize=(9, 7))

            t = np.arange(0, round((len(episode_data['i_a_training'].to_list())) * ts, 4), ts).tolist()
            v_d_ref = [325.27] * len(t)
            v_d_ref0 = [0] * len(t)



            axs[0].plot(t, (u_dq0_RL[0] * 1), 'b')
            axs[0].plot(t, (u_dq0_RL[1] * 1), 'r')
            axs[0].plot(t, (u_dq0_RL[2] * 1), 'g')
            axs[0].grid()
            if xlim:
                axs[0].set_xlim(interval_list_x)
            axs[0].set_ylabel('$u_{\mathrm{dq0,RL}}$')#\,/\,\mathrm{}$')

            axs[1].plot(t, (u_dq0[0]*1), 'b')
            axs[1].plot(t, (u_dq0[1]*1), 'r')
            axs[1].plot(t, (u_dq0[2]*1), 'g')
            axs[1].grid()
            if xlim:
                axs[1].set_xlim(interval_list_x)
            axs[1].set_ylabel('$u_{\mathrm{dq0,SG}}$')#\,/\,\mathrm{}$')

            #axs[3].plot(t, episode_data['Rewards_raw'].to_list(), 'b', label='$\mathrm{r}_\mathrm{env, unscaled}$')
            axs[2].plot(t, episode_data['Rewards_sum'].to_list(), 'r', label='$\mathrm{r}_\mathrm{punish, scaled}$')
            axs[2].grid()
            plt.legend(loc="upper right")
            axs[2].set_ylabel('r')
            if xlim:
                axs[2].set_xlim(interval_list_x)
            #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            #axs[2].text(0.25, 0.95, "$\mathrm{r}_\mathrm{min}(-0.75) \cdot (1-\gamma) = -0.04$",
             #           transform=axs[3].transAxes, fontsize=14,
              #          verticalalignment='top', bbox=props)


            axs[3].plot(t, episode_data['R_load_training'].to_list(), 'g')
            # axs[4].plot(t, episode_data['safe_guard_active'].to_list(), 'g')
            #axs[4].set_ylabel('SG-active?')
            axs[3].grid()
            if xlim:
                axs[3].set_xlim(interval_list_x)
            # axs[4].plot(t, episode_data['R_load_training'].to_list(), 'g')
            axs[3].set_ylabel('$R_{\mathrm{Load}}\,/\,\mathrm{\Omega}$')

            plt.show()
            time.sleep(0.5)


            if RLS:
                R_load = episode_data['R_load_training'].to_numpy()
                A_sys = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L + R2 / (L * (R_load + R2))],
                                  [1 / (C * (1 + R2 / R_load)), -1 / (C * (R_load + R2))]])

                # B_sys = np.array([[1 / L], [0]])

                A_sys[0, 1, :] = A_sys[0, 1, :] * v_dead / i_dead
                A_sys[1, 0, :] = A_sys[1, 0, :] * i_dead / v_dead
                # B_sys[0, 0] = B_sys[0, 0] / i_dead * v_dc

                A_d2 = np.ndarray(shape=(2, 2, len(R_load)))

                for i in range(len(R_load)):
                    A_d2[:, :, i] = scipy.linalg.expm(A_sys[:, :, i] * ts)

                ones_vec = np.ones(len(t) - 1)



                ################################ plt matrices


                if save_results:
                    params = {'backend': 'ps',
                              'text.latex.preamble': [r'\usepackage{gensymb}'
                                                      r'\usepackage{amsmath,amssymb,mathtools}'
                                                      r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                                      r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
                              'axes.labelsize': 12.5,  # fontsize for x and y labels (was 10)
                              'axes.titlesize': 12.5,
                              'font.size': 12.5,  # was 10
                              'legend.fontsize': 12.5,  # was 10
                              'xtick.labelsize': 12,
                              'ytick.labelsize': 12,
                              'text.usetex': True,
                              'figure.figsize': [8, 6],  # [4.5, 7.5],
                              'font.family': 'serif',
                              'lines.linewidth': 1.2
                              }
                    matplotlib.rcParams.update(params)

                fig, axs = plt.subplots(2, 2)#, figsize=(9, 7))
                plt.subplots_adjust(wspace = 0.4, hspace = 0.1)
                #plt.subplots_adjust(left=0.1,bottom = 0.1,right = 0.9,top = 0.9, wspace = 0.4, hspace = 0.4)
                axs[0, 0].plot(t[1:], episode_data['A11_c'].to_list()[1:], 'b', label='$A_\mathrm{RLS}$')
                axs[0, 0].plot(t[1:], A_d2[0, 0, 1:], ':', color='red', label='$A_\mathrm{}$')
                axs[0, 0].legend(bbox_to_anchor=(0.7, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                              ncol=4)
                # axs[0, 0].plot(t[1:], ones_vec * A_d[0, 0], ':', color='red')
                axs[0, 0].grid()
                if xlim:
                    axs[0, 0].set_xlim(interval_list_x)
                axs[0, 0].set_xticklabels([])
                axs[0, 0].tick_params(direction='in')
                #axs[0, 0].set_ylabel('Ad11_c')
                axs[0, 0].set_ylabel('$A_{\mathrm{11}}$')
                axs[0, 1].plot(t[1:], episode_data['A12_c'].to_list()[1:], 'b')
                axs[0, 1].plot(t[1:], A_d2[0, 1, 1:], ':', color='red')
                # axs[0, 1].plot(t[1:], ones_vec * A_d[0, 1], ':', color='red')
                axs[0, 1].grid()
                axs[0, 1].set_ylim([ -1.932, -1.918])
                if xlim:
                    axs[0, 1].set_xlim(interval_list_x)
                axs[0, 1].set_xticklabels([])
                axs[0, 1].tick_params(direction='in')
                axs[0, 1].set_ylabel('Ad12_c')
                axs[0, 1].set_ylabel('$A_{\mathrm{12}}$')
                axs[1, 0].plot(t[1:], episode_data['A21_c'].to_list()[1:], 'b')
                axs[1, 0].plot(t[1:], A_d2[1, 0, 1:], ':', color='red')
                # axs[1, 0].plot(t[1:], ones_vec * A_d[1, 0], ':', color='red')
                axs[1, 0].grid()
                if xlim:
                    axs[1, 0].set_xlim(interval_list_x)
                axs[1, 0].set_ylabel('$A_{\mathrm{21}}$')
                axs[1, 0].tick_params(direction='in')
                axs[1, 0].set_xlabel('$t_{\mathrm{}}\,/\,\mathrm{s}$')
                axs[1, 1].plot(t[1:], episode_data['A22_c'].to_list()[1:], 'b')
                axs[1, 1].plot(t[1:], A_d2[1, 1, 1:], ':', color='red')
                # axs[1, 1].plot(t[1:], ones_vec * A_d[1, 1], ':', color='red')
                axs[1, 1].grid()
                if xlim:
                    axs[1, 1].set_xlim(interval_list_x)
                axs[1, 1].tick_params(direction='in')
                axs[1, 1].set_ylabel('$A_{\mathrm{22}}$')
                axs[1, 1].set_xlabel('$t_{\mathrm{}}\,/\,\mathrm{s}$')
                #plt.tick_params(direction='in')
                plt.show()
                time.sleep(0.5)

                if save_results:
                    fig.savefig(f'{folder_name}/' + study_name + save_name + 'Ac.pgf')
                    fig.savefig(f'{folder_name}/' + study_name + save_name +'Ac.png')
                    fig.savefig(f'{folder_name}/' + study_name + save_name +'Ac.pdf')


        asd = 1