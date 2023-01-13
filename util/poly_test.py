import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from util.feasible_helper import calc_feasible_set
from util.control_utils import calc_feasible_set2

# L = 2.3e-3
# R = 400e-3
# C = 10e-6
# R_load_list = [10, 100, 200]
from util.safeguard import Safeguard

L = 70e-6
R1 = 1.1e-3
R2 = 7e-3

C = 250e-6
R_load_list = [100]  # [1, 10, 100, 1000] # @650 V 6 P_max = 250kW -> ~1.7 Ohm

i_load = 30

V_dc = 400
v_lim = 650
i_lim = 300

# V_dc = 300
# v_lim = 285
# i_lim = 16

"""
W_x = np.array([[-1, 0],
                [1, 0],
                [0, -1],
                [0, 1]])
                """
W_x = np.array([[-1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 1]
                ])

W_u = np.array([[-1, 0, 0],
                [1, 0, 0],
                [0 ,-1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1]
                ])

omega_u = np.array([V_dc, V_dc, V_dc, V_dc, V_dc, V_dc])

omega_x = np.array([i_lim, i_lim, i_lim, i_lim, i_lim, i_lim, v_lim, v_lim, v_lim, v_lim, v_lim, v_lim])

F_list = []

ts = 1e-4

for R_load in R_load_list:
    # without R_C
    # A_sys = np.array([[-R1 / L, -1 / L],
    #                  [1 / C, -1 / (C * R_load)]])
    """
    A_sys = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L - R2 / (L * (R_load + R2))],
                      [-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L - R2 / (L * (R_load + R2))],
                      [-R1 / L - R2 / (L * (1 + R2 / R_load)), -1 / L - R2 / (L * (R_load + R2))],
                      [1 / (C * (1 - R2 / R_load)), -1 / (C * (R_load + R2))],
                      [1 / (C * (1 - R2 / R_load)), -1 / (C * (R_load + R2))],
                      [1 / (C * (1 - R2 / R_load)), -1 / (C * (R_load + R2))]])
                      """

    A_sys = np.array([[-R1 / L - R2 / (L * (1 + R2 / R_load)), 0, 0, -1 / L - R2 / (L * (R_load + R2)), 0, 0],
                      [0, -R1 / L - R2 / (L * (1 + R2 / R_load)), 0, 0, -1 / L - R2 / (L * (R_load + R2)), 0],
                      [0, 0, -R1 / L - R2 / (L * (1 + R2 / R_load)), 0, 0, -1 / L - R2 / (L * (R_load + R2))],
                      [1 / (C * (1 - R2 / R_load)), 0, 0, -1 / (C * (R_load + R2)), 0, 0],
                      [0, 1 / (C * (1 - R2 / R_load)), 0, 0, -1 / (C * (R_load + R2)), 0],
                      [0, 0, 1 / (C * (1 - R2 / R_load)), 0, 0, -1 / (C * (R_load + R2))]])

    B_sys = np.array([[1 / L, 0, 0], [0 ,1 / L, 0], [0, 0, 1 / L], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    A_d = scipy.linalg.expm(A_sys * ts)
    A_inv = scipy.linalg.inv(A_sys)
    B_d = A_inv @ (A_d - np.eye(A_sys.shape[0])) @ B_sys

    feasible_set = calc_feasible_set(
        A_d, B_d, W_x, omega_x, W_u, omega_u,
        project_dim=[2, 5, 8],
        tol=0.5,
        return_all=False,
        N_max=4,
        discount=1,
        N_start=1,
        verbose=1,
        notebook_bar=False,
        progress_bar=False,
        plt_all=False
    )

    #F_list.append(F)

    # plt.title(f"R_load = {R_load}")
    # plt.show()
# F.contains(np.transpose([[600,-300,0]]))   # oder np.transpose([[600,-300,0]]) in F
# F.contains(np.array([[600],[-300],[0]]))

#safeguard = Safeguard(F)

#action_safe, sg_active = safeguard.guide(0, (0, 0))



