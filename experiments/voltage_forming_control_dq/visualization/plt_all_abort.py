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


rew_array = np.zeros([5, 149000])
rew_array_noSL = np.zeros([5, 149000])
rew_array_SL_RLS = np.zeros([5, 149000])

window_size = 1000

train_data = pd.read_pickle('data/' + meas_data_folder +'/'  + study_name + '_all_rewards' + '.pkl.bz2')

"""
4_370_vDC_noSL3_all_aborts_trial_number_4.pkl
for tr in range(5):
    #rew_array[tr, :] = train_data['all_rewards_'+str(tr)].to_numpy()
    windows = train_data['all_aborts_'+str(tr)].rolling(window_size)
    m_average = windows.mean()
    asd = m_average.to_numpy()
    rew_array[tr, :] = asd
    #rew_array[tr, :] = asd[window_size:]
"""

for tr in range(5):
    train_data_noSL = pd.read_pickle('data/' + meas_data_folder + '/' + str(4)+'_'+study_name2 +
                                     '_all_aborts_trial_number_'+str(4)+ '.pkl.bz2')
    #train_data_noSL = pd.read_pickle('data/' + meas_data_folder + '/' +  study_name2 +
        #                             '_all_aborts.pkl.bz2')
    #windows_noSL = train_data_noSL['all_aborts_'+str(tr)].rolling(window_size)
    #m_average_noSL = windows_noSL.mean()
    rew_array_noSL[tr, :] = train_data_noSL['all_aborts_'+str(tr)].to_numpy()[0:149000]

for tr in range(5):
    if tr == 0:  # take from study 1
        train_data_RLS = pd.read_pickle('data/' + meas_data_folder + '/' + str(tr) + '_' + study_RLS +
                                         #'_all_training_rewards_trial_number_' + str(tr) + '.pkl.bz2')
                                         '_all_aborts_trial_number_' + str(tr) + '.pkl.bz2')
    else: # rest in study _2

        train_data_RLS = pd.read_pickle('data/' + meas_data_folder + '/' + str(3) + '_' + study_RLS2 +
                                        # '_all_training_rewards_trial_number_' + str(tr) + '.pkl.bz2')
                                        '_all_aborts_trial_number_' + str(3) + '.pkl.bz2')
    #windows_SL_RLS = train_data_RLS['all_rewards_'+str(tr)].rolling(window_size)
    #m_average_SL_RLS = windows_SL_RLS.mean()
    rew_array_SL_RLS[tr, :] = train_data_RLS['all_aborts_'+str(tr)].to_numpy()[0:149000]


abort_mean_noSL = np.mean(rew_array_noSL, axis=0)
abort_mean_SL = np.zeros([rew_array_noSL.shape[1]])
abort_mean_RLS = np.mean(rew_array_SL_RLS, axis=0)


fig, ax = plt.subplots(figsize=(6, 3.5))
plt.plot( abort_mean_noSL, 'b', linewidth=2, label='$\mathrm{DDPG_{}}$')
plt.plot( abort_mean_SL, 'r', linewidth=2, label='$\mathrm{DDPG_{SG}}$')
plt.plot( abort_mean_RLS, 'g', linewidth=2, label='$\mathrm{DDPG_{SG,RLS}}$')

plt.ylabel('count($i_\mathrm{abc} > i_\mathrm{lim} || v_\mathrm{abc} > v_\mathrm{lim}$)')
plt.ylabel('Acc. Overcurrent/-voltage events')
#plt.ylabel('$\overline{r}_{k}$')
plt.xlabel(r'$k\mathrm{}$')
plt.tick_params(direction='in')
#plt.ylim([-1000, 11000])
plt.xlim([0, 142000])
#plt.xlim([-1, 250])
plt.grid()
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
ax.set_yscale('symlog')
ax.set_xscale('symlog')
#plt.title("DDPG with safeguard")
plt.show()

if save_results:
    fig.savefig(f'{folder_name}/all_aborts_log.pgf', bbox_inches='tight')
    fig.savefig(f'{folder_name}/all_aborts_log.png', bbox_inches='tight', dpi=500)
    fig.savefig(f'{folder_name}/all_aborts_log.pdf', bbox_inches='tight')






