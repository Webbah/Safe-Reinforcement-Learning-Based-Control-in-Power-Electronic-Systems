import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from experiments.voltage_forming_control_dq.util.config import cfg

meas_data_folder = "picard/R_load_constant/"
meas_data_folder = cfg['meas_data_folder']
study_name = cfg['STUDY_NAME']
"""
#study_name = '370_vDC_SL'
trial = ['1', '2', '3', '4']
episode_list = [list(range(68)), list(range(74)), list(range(65)), list(range(66))]
terminated_list = [[0]*68, [0]*74, [0]*65, [0]*66]
terminated_list[0] = [1 if i < 16 else tl for i, tl in enumerate(terminated_list[0]) ]
terminated_list[1] = [1 if i < 22 else tl for i, tl in enumerate(terminated_list[1]) ]
terminated_list[2] = [1 if i < 13 else tl for i, tl in enumerate(terminated_list[2]) ]
terminated_list[3] = [1 if i < 14 else tl for i, tl in enumerate(terminated_list[3]) ]
"""


trial = ['0']
episode_list = [list(range(54))]
terminated_list = [[0]*54]
terminated_list[0] = [1 if i < 2 else tl for i, tl in enumerate(terminated_list[0]) ]
#terminated_list[0][12] = 1
#terminated_list[0][14] = 1
#terminated_list[0][30] = 1


#study_name = '370_vDC_noSL'
#trial = ['2', '3', '4', '5', '6']

#episode_list = ['55', '55']
#terminated_list = ['0', '0']
#study_name = '370_vDC_SL_RLS'
#trial = ['2']



rew_array = np.zeros([len(trial), 150000])
eps_len_array = np.array([])

for tr in range(len(trial)):
    cc = 0
    for episode, terminated in zip(episode_list[tr], terminated_list[tr]):
        episode_data = pd.read_pickle('data/' + meas_data_folder + '/'+study_name+ '/' + trial[tr]+ '/' + trial[tr] + '_' + study_name +
                                      '_training_episode_number_' + str(episode) + '_terminated' + str(terminated)  + '.pkl.bz2')


        rew = episode_data['Rewards_sum'].to_numpy()


        rew_array[tr,cc:len(rew)+cc] = rew
        cc = len(rew)+cc
        asd = 1
        #eps_len_array[tr,:] = train_data['Num_steps_per_episode'].to_numpy()

#itemindex = np.where(rew[0:,] == -1)


abbruch = np.zeros([len(trial), 150000])



for ttt in range(len(trial)):
    fail = 0
    itemindex = np.where(rew_array[ttt, :] == -1)
    for i in range(150000):
        if i in itemindex[0]:
            fail +=1
        abbruch[ttt, i] = fail

#np.append(rew_array, train_data['Mean_eps_reward_sum'].to_numpy(), axis=1)

asd = 1

train_data = {
            "all_aborts_0": abbruch[0,:]
            #"all_aborts_2": abbruch[1,:],
            #"all_aborts_3": abbruch[2,:],
            #"all_aborts_4": abbruch[3,:]
        }

meas_data_folder = "rewards/"
df = pd.DataFrame(train_data)
df.to_pickle('data/' +meas_data_folder +  str(tr) + '_'+ study_name +'2222_all_aborts_trial_number_' + str(tr) + '.pkl.bz2')
