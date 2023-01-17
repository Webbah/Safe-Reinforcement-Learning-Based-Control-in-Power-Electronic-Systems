import itertools
import json
import os
import platform
import sys
import os

module_path = os.path.abspath(os.getcwd())

if module_path not in sys.path:

    sys.path.append(module_path)

import numpy as np
import optuna
from optuna.samplers import TPESampler
from experiments.voltage_forming_control_dq.train_agent import train_ddpg
from experiments.voltage_forming_control_dq.util.config import cfg
from util.lr_scheduler import linear_schedule

STUDY_NAME = cfg['STUDY_NAME']

node = platform.uname().node



def ddpg_objective_fix_params_optuna(trial):
    #file_congfig = open('experiments/voltage_forming_control_dq/Best_P10.json', )
    file_congfig = open('experiments/voltage_forming_control_dq/PC2_DDPG_Vctrl_single_inv_22_newTestcase_Trial_number_11534_0.json', )
    trial_config = json.load(file_congfig)

    number_learning_steps = 150000  # trial.suggest_int("number_learning_steps", 100000, 1000000)
    # rew_weigth = trial.suggest_float("rew_weigth", 0.1, 5)
    # rew_penalty_distribution = trial.suggest_float("antiwindup_weight", 0.1, 5)
    penalty_I_weight = trial_config["penalty_I_weight"]  # trial.suggest_float("penalty_I_weight", 100e-6, 2)
    penalty_P_weight = trial_config["penalty_P_weight"]  # trial.suggest_float("penalty_P_weight", 100e-6, 2)

    penalty_I_decay_start = trial_config[
        "penalty_I_decay_start"]  # trial.suggest_float("penalty_I_decay_start", 0.00001, 1)
    penalty_P_decay_start = trial_config[
        "penalty_P_decay_start"]  # trial.suggest_float("penalty_P_decay_start", 0.00001, 1)

    t_start_penalty_I = int(penalty_I_decay_start * number_learning_steps)
    t_start_penalty_P = int(penalty_P_decay_start * number_learning_steps)

    integrator_weight = trial_config["integrator_weight"]  # trial.suggest_float("integrator_weight", 1 / 200, 2)
    # integrator_weight = trial.suggest_loguniform("integrator_weight", 1e-6, 1e-0)
    # antiwindup_weight = trial.suggest_loguniform("antiwindup_weight", 50e-6, 50e-3)
    antiwindup_weight = trial_config["antiwindup_weight"]  # trial.suggest_float("antiwindup_weight", 0.00001, 1)

    learning_rate = trial_config["learning_rate"]  # trial.suggest_loguniform("learning_rate", 1e-6, 1e-1)  # 0.0002#

    lr_decay_start = trial_config[
        "lr_decay_start"]  # trial.suggest_float("lr_decay_start", 0.00001, 1)  # 3000  # 0.2 * number_learning_steps?
    lr_decay_duration = trial_config["lr_decay_duration"]  # trial.suggest_float("lr_decay_duration", 0.00001,
    #  1)  # 3000  # 0.2 * number_learning_steps?
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))
    final_lr = trial_config["final_lr"]  # trial.suggest_float("final_lr", 0.00001, 1)

    gamma = trial_config["gamma"]  # trial.suggest_float("gamma", 0.5, 0.9999)
    weight_scale = trial_config["weight_scale"]  # trial.suggest_loguniform("weight_scale", 5e-5, 0.2)  # 0.005

    bias_scale = trial_config["bias_scale"]  # trial.suggest_loguniform("bias_scale", 5e-4, 0.1)  # 0.005
    alpha_relu_actor = trial_config[
        "alpha_relu_actor"]  # trial.suggest_loguniform("alpha_relu_actor", 0.0001, 0.5)  # 0.005
    alpha_relu_critic = trial_config[
        "alpha_relu_critic"]  # trial.suggest_loguniform("alpha_relu_critic", 0.0001, 0.5)  # 0.005

    batch_size = trial_config["batch_size"]  # trial.suggest_int("batch_size", 16, 1024)  # 128
    buffer_size = trial_config[
        "buffer_size"]  # trial.suggest_int("buffer_size", int(1e4), number_learning_steps)  # 128

    actor_hidden_size = trial_config[
        "actor_hidden_size"]  # trial.suggest_int("actor_hidden_size", 10, 200)  # 100  # Using LeakyReLU
    actor_number_layers = trial_config["actor_number_layers"]  # trial.suggest_int("actor_number_layers", 1, 4)

    critic_hidden_size = trial_config["critic_hidden_size"]  # trial.suggest_int("critic_hidden_size", 10, 300)  # 100
    critic_number_layers = trial_config["critic_number_layers"]  # trial.suggest_int("critic_number_layers", 1, 4)

    n_trail = str(trial.number)
    use_gamma_in_rew = 1
    noise_var = trial_config["noise_var"]  # trial.suggest_loguniform("noise_var", 0.01, 1)  # 2
    # min var, action noise is reduced to (depends on noise_var)
    noise_var_min = 0.0013  # trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
    # min var, action noise is reduced to (depends on training_episode_length)
    noise_steps_annealing = int(
        0.25 * number_learning_steps)  # trail.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
    # number_learning_steps)
    noise_theta = trial_config["noise_theta"]  # trial.suggest_loguniform("noise_theta", 1, 50)  # 25  # stiffness of OU
    error_exponent = 0.5  # trial.suggest_loguniform("error_exponent", 0.001, 4)

    training_episode_length = trial_config["training_episode_length"]
    # learning_starts = 0.32  # trial.suggest_loguniform("learning_starts", 0.1, 2)  # 128
    tau = trial_config["tau"]  # trial.suggest_loguniform("tau", 0.0001, 0.3)  # 2

    train_freq_type = "step"  # trial.suggest_categorical("train_freq_type", ["episode", "step"])
    train_freq = trial_config["train_freq"]  # trial.suggest_int("train_freq", 1, 15000)

    optimizer = trial_config[
        "optimizer"]  # trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])  # , "LBFGS"])

    number_past_vals = 2#trial_config["number_past_vals"]  # trial.suggest_int("number_past_vals", 0, 15)
    #number_past_vals = trial.suggest_int("number_past_vals", 0, 7)

    number_learning_steps = trial.suggest_int("number_learning_steps", number_learning_steps, number_learning_steps)
    seed = trial.suggest_int("seed", 0, 10)
    actor_hidden_size = trial.suggest_int("actor_hidden_size", actor_hidden_size, actor_hidden_size)  # Using LeakyReLU
    actor_number_layers = trial.suggest_int("actor_number_layers", actor_number_layers, actor_number_layers)
    alpha_relu_actor = trial.suggest_loguniform("alpha_relu_actor", alpha_relu_actor, alpha_relu_actor)
    alpha_relu_critic = trial.suggest_loguniform("alpha_relu_critic", alpha_relu_critic, alpha_relu_critic)
    antiwindup_weight = trial.suggest_float("antiwindup_weight", antiwindup_weight, antiwindup_weight)
    batch_size = trial.suggest_int("batch_size", batch_size, batch_size)
    bias_scale = trial.suggest_loguniform("bias_scale", bias_scale, bias_scale)
    buffer_size = trial.suggest_int("buffer_size", buffer_size, buffer_size)  # 128
    critic_hidden_size = trial.suggest_int("critic_hidden_size", critic_hidden_size, critic_hidden_size)
    critic_number_layers = trial.suggest_int("critic_number_layers", critic_number_layers, critic_number_layers)
    error_exponent = 0.5  # 0.5  # trial.suggest_loguniform("error_exponent", 0.001, 4)
    final_lr = trial.suggest_float("final_lr", final_lr, final_lr)
    gamma = trial.suggest_float("gamma", gamma, gamma)
    integrator_weight = trial.suggest_float("integrator_weight", integrator_weight, integrator_weight)
    learning_rate = trial.suggest_loguniform("learning_rate", learning_rate, learning_rate)
    #learning_rate = 0.00003
    lr_decay_start = trial.suggest_float("lr_decay_start", lr_decay_start, lr_decay_start)
    lr_decay_duration = trial.suggest_float("lr_decay_duration", lr_decay_duration, lr_decay_duration)
    n_trail = str(trial.number)
    #noise_steps_annealing = trial.suggest_int("noise_steps_annealing", int(0.1 * number_learning_steps),
    #                                          number_learning_steps)
    noise_theta = trial.suggest_loguniform("noise_theta", noise_theta, noise_theta)  # 25  # stiffness of OU
    noise_var = trial.suggest_loguniform("noise_var", noise_var, noise_var)  # 2
    noise_var_min = 0.0013  # trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
    number_past_vals = trial.suggest_int("number_past_vals", number_past_vals, number_past_vals)
    #optimizer = trial.suggest_categorical("optimizer", [optimizer])  # , "LBFGS"])
    optimizer = trial.suggest_categorical("optimizer", [optimizer])  # , "LBFGS"])
    penalty_I_weight = trial.suggest_float("penalty_I_weight", penalty_I_weight, penalty_I_weight)
    penalty_P_weight = trial.suggest_float("penalty_P_weight", penalty_P_weight, penalty_P_weight)

    penalty_I_decay_start = trial.suggest_float("penalty_I_decay_start", penalty_I_decay_start, penalty_I_decay_start)
    penalty_P_decay_start = trial.suggest_float("penalty_P_decay_start", penalty_P_decay_start, penalty_P_decay_start)

    t_start_penalty_I = int(penalty_I_decay_start * number_learning_steps)
    t_start_penalty_P = int(penalty_P_decay_start * number_learning_steps)
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))
    tau = trial.suggest_loguniform("tau", tau, tau)  # 2
    train_freq_type = "step"  # trial.suggest_categorical("train_freq_type", ["episode", "step"])
    training_episode_length = trial.suggest_int("training_episode_length", training_episode_length, training_episode_length)  # 128
    #training_episode_length = trial.suggest_int("training_episode_length", 100, 100)  # 128
    train_freq = trial.suggest_int("train_freq", train_freq, train_freq)
    use_gamma_in_rew = 1
    weight_scale = trial.suggest_loguniform("weight_scale", weight_scale, weight_scale)

    learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                    t_start=t_start,
                                    t_end=t_end,
                                    total_timesteps=number_learning_steps)

    safe_layer = trial.suggest_int("safe_layer", 1, 1)
    learn_model = trial.suggest_int("learn_model", 1, 1)

    loss = train_ddpg(True, learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                      alpha_relu_actor,
                      batch_size,
                      actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                      alpha_relu_critic,
                      noise_var, noise_theta, error_exponent,
                      training_episode_length, buffer_size,  # learning_starts,
                      tau, number_learning_steps, integrator_weight,
                      integrator_weight * antiwindup_weight, penalty_I_weight, penalty_P_weight,
                      train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
                      number_past_vals, seed, n_trail, safe_layer, learn_model)

    """

    loss = execute_ddpg(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                      alpha_relu_actor,
                      batch_size,
                      actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                      alpha_relu_critic,
                      noise_var, noise_theta, error_exponent,
                      training_episode_length, buffer_size,  # learning_starts,
                      tau, number_learning_steps, integrator_weight,
                      integrator_weight * antiwindup_weight, penalty_I_weight, penalty_P_weight,
                      train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
                      number_past_vals, n_trail)
    """
    return loss

def optuna_optimize_sqlite(objective, sampler=None, study_name='dummy'):

    n_trials = 10

    print(n_trials)
    print('Local optimization is run but measurement data is logged workingdirectory/data!')

    if node in cfg['lea_vpn_nodes']:
        optuna_path = './optuna/'
    else:
        # assume we are on not of pc2 -> store to project folder
        optuna_path = '/scratch/hpc-prf-reinfl/weber/OMG/optuna/'

    os.makedirs(optuna_path, exist_ok=True)

    study = optuna.create_study(study_name=study_name,
                                direction='maximize',
                                storage=f'sqlite:///{optuna_path}optuna.sqlite',
                                load_if_exists=True,
                                sampler=sampler
                                )
    #study.optimize(objective, n_trials=n_trials)
    study.optimize(objective)

if __name__ == '__main__':
    #learning_rate = list(itertools.chain(*[[1e-9] * 1]))
    #seed = [0, 1, 2]
    #number_past_vals = [0, 1, 2, 3, 4, 5, 6, 7]
    seed = [1]
    number_past_vals = [2]
    search_space = {'seed': seed, 'number_past_vals': number_past_vals}  # , 'number_learning_steps': number_learning_steps}


    #loss = ddpg_objective_fix_params()
    #print(loss)

    #TPE_sampler = TPESampler(n_startup_trials=1000)  # , constant_liar=True)

    #optuna_optimize_mysql_lea35(ddpg_objective_fix_params_optuna, study_name=STUDY_NAME, sampler=TPE_sampler)
    #optuna_optimize_sqlite(ddpg_objective, study_name=STUDY_NAME, sampler=TPE_sampler)
    optuna_optimize_sqlite(ddpg_objective_fix_params_optuna, study_name=STUDY_NAME,
                           sampler=optuna.samplers.GridSampler(search_space))