import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from experiments.voltage_forming_control_dq.util.config import cfg

meas_data_folder = "picard/R_load_constant/"
meas_data_folder = cfg['meas_data_folder']


study_name = cfg['STUDY_NAME']
#study_name = '370_vDC_SL'
trial = ['1', '2', '3', '4']
episode_list = [list(range(68)), list(range(74)), list(range(65)), list(range(66))]
terminated_list = [[0]*68, [0]*74, [0]*65, [0]*66]
terminated_list[0] = [1 if i < 16 else tl for i, tl in enumerate(terminated_list[0]) ]
terminated_list[1] = [1 if i < 22 else tl for i, tl in enumerate(terminated_list[1]) ]
terminated_list[2] = [1 if i < 13 else tl for i, tl in enumerate(terminated_list[2]) ]
terminated_list[3] = [1 if i < 14 else tl for i, tl in enumerate(terminated_list[3]) ]


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


#np.append(rew_array, train_data['Mean_eps_reward_sum'].to_numpy(), axis=1)

train_data = {
            "all_rewards_1": rew_array[0,:],
            "all_rewards_2": rew_array[1,:],
            "all_rewards_3": rew_array[2,:],
            "all_rewards_4": rew_array[3,:]
        }

df = pd.DataFrame(train_data)
df.to_pickle('data/' +meas_data_folder +  str(tr) + '_'+ study_name +'_all_rewards_trial_number_' + str(tr) + '.pkl.bz2')
