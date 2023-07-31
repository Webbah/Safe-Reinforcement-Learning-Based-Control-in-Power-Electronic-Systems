import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from experiments.voltage_forming_control_dq.envs.rewards import Reward
from experiments.voltage_forming_control_dq.envs.vctrl_single_inv import net

reward = Reward(net.v_nom, net['inverter1'].v_lim, net['inverter1'].v_DC, 0, det_run=True,
                 use_gamma_normalization=1, error_exponent=0.5, i_lim=net['inverter1'].i_lim,
                 i_nom=net['inverter1'].i_nom)

resolution = 400
SP = np.linspace(-1.1*net.v_nom/net['inverter1'].v_lim, 1.1*net.v_nom/net['inverter1'].v_lim, resolution)
meas = np.linspace(-1.2, 1.2, resolution)
i_meas = np.array([0.1])

pos_mat, vel_mat = np.meshgrid(SP, meas)
rew = np.zeros([resolution, resolution])

for pos_idx, s in enumerate(SP):
    for vel_idx, m in enumerate(meas):
        #rew[vel_idx, pos_idx] = pos*vel#reward(np.array([pos, vel]))
        rew[vel_idx, pos_idx] = reward.rew_fun_dq0_raw(np.array([m]), np.array([s]), i_meas)


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


# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(pos_mat, vel_mat, rew, cmap="viridis")
#ax.scatter(visited_states[:, 0], visited_states[:, 1], -visited_values - 0.05, color="red")
ax.set_xlabel('\n\n$v^*/v_{\mathrm{lim}}$')
ax.set_zlim([-1,1])
ax.set_ylabel('\n\n$v_\mathrm{c}/v_{\mathrm{lim}}$')
ax.set_zlabel(r'$r_\mathrm{}$', rotation=45)
ax.view_init(20, 25)
plt.show()

save_results = True
folder_name = "plots_paper"
if save_results:
    fig.savefig(f'{folder_name}/Reward_v_mre.pgf')
    fig.savefig(f'{folder_name}/Reward_v_mre.png')
    fig.savefig(f'{folder_name}/Reward_v_mre.pdf')
    fig.savefig(f'{folder_name}/Reward_v_mre.svg', format='svg', dpi=1200)