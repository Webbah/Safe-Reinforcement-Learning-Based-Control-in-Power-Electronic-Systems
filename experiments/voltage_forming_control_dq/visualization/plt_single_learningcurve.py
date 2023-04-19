import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from openmodelica_microgrid_gym.util import dq0_to_abc
from feasible_set_test import feasible_set, A_d, B_d, v_dead, i_dead

from experiments.voltage_forming_control_dq.util.config import cfg

meas_data_folder = "rewards"
#meas_data_folder = "picard/Rvar_RLS/"


study_name = '370_vDC_SL'
study_name2 = '370_vDC_noSL3'
study_RLS = '370_vDC_SL_RLS_Rconst'
study_RLS2 = '370_vDC_SL_RLS_Rconst_2'
study_RLS_picard = '370_vDC_RLS_constR'

save_results = True
folder_name = "plots_paper"

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


rew_array = np.zeros([5, 150000])
rew_array_noSL = np.zeros([5, 151000])
rew_array_SL_RLS = np.zeros([5, 150000])

window_size = 1000

train_data = pd.read_pickle('data/' + meas_data_folder +'/'  + study_name + '_all_rewards' + '.pkl.bz2')


for tr in range(5):
    #rew_array[tr, :] = train_data['all_rewards_'+str(tr)].to_numpy()
    windows = train_data['all_rewards_'+str(tr)].rolling(window_size)
    m_average = windows.mean()
    asd = m_average.to_numpy()
    rew_array[tr, :] = asd
    #rew_array[tr, :] = asd[window_size:]

for tr in range(5):
    #train_data_noSL = pd.read_pickle('data/' + meas_data_folder + '/' + str(tr)+'_'+study_name2 +
     #                                '_all_training_rewards_trial_number_'+str(tr)+ '.pkl.bz2')
    train_data_noSL = pd.read_pickle('data/' + meas_data_folder + '/' +  study_name2 +
                                     '_all_rewards.pkl.bz2')
    windows_noSL = train_data_noSL['all_rewards_'+str(tr)].rolling(window_size)
    m_average_noSL = windows_noSL.mean()
    rew_array_noSL[tr, :] = m_average_noSL.to_numpy()

for tr in range(5):
    if tr == 0:  # take from study 1
        train_data_RLS = pd.read_pickle('data/' + meas_data_folder + '/' + str(tr) + '_' + study_RLS +
                                         #'_all_training_rewards_trial_number_' + str(tr) + '.pkl.bz2')
                                         '_all_rewards_trial_number_' + str(tr) + '.pkl.bz2')
    else: # rest in study _2

        train_data_RLS = pd.read_pickle('data/' + meas_data_folder + '/' + str(3) + '_' + study_RLS2 +
                                        # '_all_training_rewards_trial_number_' + str(tr) + '.pkl.bz2')
                                        '_all_rewards_trial_number_' + str(3) + '.pkl.bz2')
    windows_SL_RLS = train_data_RLS['all_rewards_'+str(tr)].rolling(window_size)
    m_average_SL_RLS = windows_SL_RLS.mean()
    rew_array_SL_RLS[tr, :] = m_average_SL_RLS.to_numpy()

    # picard
    #train_data_RLS = pd.read_pickle('data/' + meas_data_folder + '/' + study_RLS_picard + '_all_rewards_picard_0' + '.pkl.bz2')
    #windows_SL_RLS = train_data_RLS['all_rewards_' + str(tr)].rolling(window_size)
    #m_average_SL_RLS = windows_SL_RLS.mean()
    #rew_array_SL_RLS[tr+1, :] = m_average_SL_RLS[:150000].to_numpy()



#rew_array = train_data['Mean_eps_reward_sum'].to_numpy()
#episode = np.array([list(range(0, len(rew_array)))]).squeeze()
rew_mean = np.mean(rew_array, axis=0)
rew_std = np.std(rew_array, axis=0)

rew_mean_noSL = np.mean(rew_array_noSL, axis=0)
rew_std_noSL = np.std(rew_array_noSL, axis=0)

rew_mean_RLS = np.mean(rew_array_SL_RLS, axis=0)
rew_std_RLS = np.std(rew_array_SL_RLS, axis=0)

stop = 144000

rew_mean_plt = rew_mean[window_size:stop]
rew_std_plt = rew_std[window_size:stop]

rew_mean_plt_noSL = rew_mean_noSL[window_size:stop]
rew_std_plt_noSL = rew_std_noSL[window_size:stop]

rew_mean_plt_RLS = rew_mean_RLS[window_size:stop]
rew_std_plt_RLS = rew_std_RLS[window_size:stop]

steps = np.arange(len(rew_mean_plt))

fig, ax = plt.subplots(figsize=(6, 3.5))
"""
plt.fill_between(steps, rew_mean_plt_noSL + rew_std_plt_noSL, rew_mean_plt_noSL - rew_std_plt_noSL, facecolor='b', alpha=0.25)
plt.plot( rew_mean_plt_noSL, 'b', linewidth=2, label='$\mathrm{DDPG_{}}$')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
plt.plot(rew_mean_plt_noSL - rew_std_plt_noSL, '--b', linewidth=0.5)
plt.plot(rew_mean_plt_noSL + rew_std_plt_noSL, '--b', linewidth=0.5)
"""
#plt.fill_between(steps, rew_mean_plt + rew_std_plt, rew_mean_plt - rew_std_plt, facecolor='r', alpha=0.25)
#plt.plot( rew_mean_plt, 'r', linewidth=2, label='$\mathrm{DDPG_{SG}}$')
#plt.plot( rew_array[0,window_size:stop], 'r', linewidth=1, label='$\mathrm{DDPG_{SG}}$')
plt.plot( rew_array[1,window_size:stop], 'r', linewidth=2, label='$\mathrm{DDPG_{SG}}$')
#plt.plot( rew_array[2,window_size:stop], 'r', linewidth=1, label='$\mathrm{DDPG_{SG}}$')
#plt.plot( rew_array[3,window_size:stop], 'r', linewidth=1, label='$\mathrm{DDPG_{SG}}$')
#plt.plot( rew_array[4,window_size:stop], 'r', linewidth=1, label='$\mathrm{DDPG_{SG}}$')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
#plt.plot(rew_mean_plt - rew_std_plt, '--r', linewidth=0.5)
#plt.plot(rew_mean_plt + rew_std_plt, '--r', linewidth=0.5)
"""
plt.fill_between(steps, rew_mean_plt_RLS + rew_std_plt_RLS, rew_mean_plt - rew_std_plt_RLS, facecolor='g', alpha=0.25)
plt.plot( rew_mean_plt_RLS, 'g', linewidth=2, label='$\mathrm{DDPG_{SG,RLS}}$')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
plt.plot(rew_mean_plt_RLS - rew_std_plt_RLS, '--g', linewidth=0.5)
plt.plot(rew_mean_plt_RLS + rew_std_plt_RLS, '--g', linewidth=0.5)
"""
#plt.plot( rew_mean_plt_noSL, 'r', linewidth=2)#, label='$\mathrm{SEC}$')
#plt.plot( rew_mean[window_size:], 'b', linewidth=2)#, label='$\mathrm{SEC}$')
#plt.ylim([0.034, 0.049])
#plt.plot(episode, rew_mean, 'b', linewidth=2)#, label='$\mathrm{SEC}$')

plt.ylabel('$\overline{r}_{k}$')
plt.xlabel(r'$k\mathrm{}$')
plt.tick_params(direction='in')
#plt.ylim([-0.025, 0.055])
plt.xlim([-500, 143300])
plt.grid()
#plt.title("DDPG with safeguard")
plt.show()
"""
plt.fill_between(steps, rew_mean_plt_RLS + rew_std_plt_RLS, rew_mean_plt - rew_std_plt_RLS, facecolor='g', alpha=0.25)
plt.plot( rew_mean_plt_RLS, 'g', linewidth=2, label='$\mathrm{DDPG_{SG,RLS}}$')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
plt.plot(rew_mean_plt_RLS - rew_std_plt_RLS, '--g', linewidth=0.5)
plt.plot(rew_mean_plt_RLS + rew_std_plt_RLS, '--g', linewidth=0.5)
plt.show()
"""

if save_results:
    fig.savefig(f'{folder_name}/rewards_SG.pgf', bbox_inches='tight')
    fig.savefig(f'{folder_name}/rewards_SG.png', bbox_inches='tight', dpi=500)
    fig.savefig(f'{folder_name}/rewards_SG.pdf', bbox_inches='tight')






