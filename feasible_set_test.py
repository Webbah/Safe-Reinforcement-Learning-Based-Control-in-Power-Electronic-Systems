import matplotlib.pyplot as plt
import numpy as np
import pypoman
import scipy.linalg

from experiments.voltage_forming_control_dq.envs.vctrl_single_inv import L_filter, C_filter, R_load_fixed, R_filter_C, \
    R_filter, net
#from util.feasible_helper import calc_feasible_set
from util.control_utils import calc_feasible_set2
from util.feasible_set_iterative import calc_feasible_set, plot_polytope_3d

# L = 2.3e-3
# R = 400e-3
# C = 10e-6
# R_load_list = [10, 100, 200]
from util.safeguard import Safeguard

#L = 70e-6
R1 = 1.1e-3
R2 = 7e-3
C = 250e-6
R_load_list = [100]  # [1, 10, 100, 1000] # @650 V 6 P_max = 250kW -> ~1.7 Ohm

L = L_filter
R1 = R_filter
R2 = R_filter_C
C = C_filter
R_load_list = [R_load_fixed]  # [1, 10, 100, 1000] # @650 V 6 P_max = 250kW -> ~1.7 Ohm

i_load = 30

#V_dc = 50
#v_nom = 40
#v_lim = 1.2*v_nom   # v_Dead = 650, but should not increase 390 V! (+20% v_ref)
#i_nom = 30  # i_dead = 400 A - 300-> i_nom
#v_dead = 60  # net -> v_lim
#i_dead = 40  # net -> i_lim

i_dead= net['inverter1'].i_lim  # inverter current limit / A
i_nom = net['inverter1'].i_nom  # nominal inverter current / A
v_nom = net.v_nom
v_lim = 1.2 * v_nom
v_dead = net['inverter1'].v_lim
v_dc = net['inverter1'].v_DC/2    # because f in OME is /2

# V_dc = 300
# v_lim = 285
# i_lim = 16

normalized = True

W_x = np.array([[-1, 0],
                [1, 0],
                [0, -1],
                [0, 1]])

W_u = np.array([[-1],
                [1],
                ])

if not normalized:
    omega_u = np.array([v_dc, v_dc])
    omega_x = np.array([i_nom, i_nom, v_lim, v_lim])
else:
    # normalized - if Ad and Bd are used they have to be normalized aas well!!!
    omega_u = np.array([1, 1])  # since the action u=1 is multiplied in modelica by Vdc
    omega_x = np.array([i_nom/i_dead, i_nom/i_dead, v_lim/v_dead, v_lim/v_dead])

#omega_u = np.array([1, 1])
#omega_x = np.array([1, 1, 1, 1])

F_list = []

ts = 1e-4

for R_load in R_load_list:
    # without R_C
    # A_sys = np.array([[-R1 / L, -1 / L],
    #                  [1 / C, -1 / (C * R_load)]])
    # x = [i, v]
    A_sys = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L + R2 / (L * (R_load + R2))],
                      [1 / (C * (1 + R2 / R_load)), -1 / (C * (R_load + R2))]])

    B_sys = np.array([[1 / L], [0]])

    if normalized:
        A_sys[0, 1] = A_sys[0, 1] * v_dead / i_dead
        A_sys[1, 0] = A_sys[1, 0] * i_dead / v_dead
        B_sys[0, 0] = B_sys[0, 0] / i_dead * v_dc

    A_d = scipy.linalg.expm(A_sys * ts)
    A_inv = scipy.linalg.inv(A_sys)
    B_d = A_inv @ (A_d - np.eye(A_sys.shape[0])) @ B_sys






    feasible_set = calc_feasible_set(
        A_d, B_d, W_x, omega_x, W_u, omega_u,
        project_dim=[1, 2, 3],
        return_all=False,
        N_max=1,    # todo change for better polytope function, 1 iteration since matlab ok, but validate!!!
        N_start=1,
    )

    #F_list.append(F)

    # plt.title(f"R_load = {R_load}")
    # plt.show()
# F.contains(np.transpose([[600,-300,0]]))   # oder np.transpose([[600,-300,0]]) in F
# F.contains(np.array([[600],[-300],[0]]))
"""
plot_polytope_3d(feasible_set)
vertices = pypoman.compute_polytope_vertices(feasible_set.A, feasible_set.b)
#safeguard = Safeguard(feasible_set)

#action_safe, sg_active = safeguard.guide(0, (0, 0))
F_v_a = feasible_set.project([2, 3])
fig, axis = plt.subplots(1, 1, figsize=(8, 4))
F_v_a.plot(axis, color='orange')
axis.set_xlim(-0.8, 0.8)
axis.set_ylim(-1.2, 1.2)
axis.set_ylabel('$m_{}$')
axis.set_xlabel('$v_{C}$')
axis.grid(True)
plt.show()

F_i_a = feasible_set.project([1, 3])
fig, axis = plt.subplots(1, 1, figsize=(8, 4))
F_i_a.plot(axis, color='orange')
axis.set_xlim(-0.8, 0.8)
axis.set_ylim(-1.2, 1.2)
axis.set_ylabel('$m_{}$')
axis.set_xlabel('$i_{L}$')
axis.grid(True)
plt.show()
sad = 1
"""

"""
R_load2 = 50
A_sys2 = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load2)), -1 / L + R2 / (L * (R_load2 + R2))],
                  [1 / (C * (1 + R2 / R_load2)), -1 / (C * (R_load2 + R2))]])

B_sys2 = np.array([[1 / L], [0]])

A_d2 = scipy.linalg.expm(A_sys2 * ts)
A_inv2 = scipy.linalg.inv(A_sys2)
B_d2 = A_inv2 @ (A_d2 - np.eye(A_sys2.shape[0])) @ B_sys2

feasible_set2 = calc_feasible_set(
    A_d2, B_d2, W_x, omega_x, W_u, omega_u,
    project_dim=[1, 2, 3],
    tol=0.5,
    return_all=False,
    N_max=1,  # todo change for better polytope function, 1 iteration since matlab ok, but validate!!!
    discount=1,
    N_start=1,
    verbose=1,
    notebook_bar=False,
    progress_bar=False,
    plt_all=False
)
"""