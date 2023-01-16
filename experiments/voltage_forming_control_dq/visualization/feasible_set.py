import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, LinearConstraint
import polytope as poly
import control
import pypoman
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#from util.config import state_space_system_LC_normalized, ts

from feasible_set_test import W_u, W_x, omega_u, omega_x, A_d, B_d, ts, v_lim, v_dead, i_dead, i_nom
from util.arrow_3d import _arrow3D
from util.feasible_set_iterative import calc_feasible_set
from util.safeguard import Safeguard

N = 1
"""
feasible_set = calc_feasible_set(
        A_d, B_d, W_x, omega_x, W_u, omega_u,
        project_dim=[1, 2, 3],
        return_all=False,
        N_max=1,    # todo change for better polytope function, 1 iteration since matlab ok, but validate!!!
        N_start=1,
    )
    """
feasible_set = calc_feasible_set(A=A_d,
                                 B=B_d,
                                 Wx=W_x,
                                 omega_x=omega_x,
                                 Wu=W_u,
                                 omega_u=omega_u,
                                 project_dim=[1, 2, 3],
                                 return_all=False,
                                 N_max=N,    # todo change for better polytope function, 1 iteration based on matlab ok!!! (-> Scipy?)
                                 )

vertices = pypoman.compute_polytope_vertices(feasible_set.A, feasible_set.b)
sg_sc = Safeguard(constraints=feasible_set, set_scaler=0, state_limits=np.array([i_nom/i_dead, v_lim/v_dead]),
          action_limits=np.array([1]),
          A_d=A_d,
          B_d=B_d, ts=ts)#,        soft_constraint=True )


#unsafe_state_points_v_sc = [(0.8, -0.5), (0.1, -0.65), (1.1, -0.5),]
#actions_v_sc = [0.95, 0.8, 0.1]

unsafe_state_points_v_sc = [(0, 0), (0.7500, 0.0970), (0.7431, 0.2911), (0.7531, 0.4853), (0.1842, 0.6050),
                            (-0.1554, 0.6057), (-0.0782, 0.5714), (0.0371, 0.5631), (0.1373, 0.5836)]

actions_v_sc = [3, 3, 3, 3, 3, 3, 3, 3]





v_lim = v_lim/v_dead
i_lim = i_nom/i_dead



u_safe_v_sc = [sg_sc.guide(u, x)[0][0] for u, x in zip(actions_v_sc, unsafe_state_points_v_sc)]




F_v_a = feasible_set.project([2, 3])
fig, axis = plt.subplots(1, 1, figsize=(8, 4))
F_v_a.plot(axis, color='orange')
plt.plot([-v_lim, -v_lim], [-1, 1], 'r', linestyle="--", label="$v_\mathrm{lim}$")
plt.plot([v_lim, v_lim], [-1, 1], 'r', linestyle="--")
plt.plot([-v_lim, v_lim], [-1, -1], 'r', linestyle="-.", label="$m_\mathrm{lim}$")
plt.plot([-v_lim, v_lim], [1, 1], 'r', linestyle="-.")
axis.grid(True)
plt.plot(unsafe_state_points_v_sc[0][1], actions_v_sc[0], color='blue', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{RL}$")
plt.plot(unsafe_state_points_v_sc[0][1], u_safe_v_sc[0], color='green', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{SG}$")
for u_sg, u_rl, x in zip(u_safe_v_sc, actions_v_sc, unsafe_state_points_v_sc):
    plt.plot(x[1], u_rl, color='blue', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{RL}$")
    plt.plot(x[1], u_sg, color='green', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{SG}$")
    plt.arrow(x[1], u_rl, 0, u_sg - u_rl -0.1*np.sign(u_sg-u_rl)  , color='red', head_length = 0.07, head_width = 0.025, length_includes_head = True)
#axis.set_xlim(-1.2, 1.2)
#axis.set_ylim(-1.2, 1.2)
axis.legend()
axis.set_ylabel('$m_{}$')
axis.set_xlabel('$v_{C}$')
plt.show()


sg = Safeguard(constraints=feasible_set, set_scaler=0, state_limits=np.array([300 / 400, 325 * 1.2 / 650]),
          action_limits=np.array([1]),
          A_d=A_d,
          B_d=B_d, ts=ts,
         )


unsafe_state_points_v = [(0.7, -0.5)]#, (0.1, -0.4), (0.5, -0.2), (-0.6, 0.55), (-0.4, 0.3)]
actions_v = [0.95]#, 0.8, 0.9, -0.8, -0.6]
unsafe_state_points_i = [(-0.6, -0.6)]#, (-0.7, -0.5), (0.7, 0.55)]
actions_i = [-0.95]#, -0.8, 0.9]

#u_safe_v, sg_active_v = sg.guide(actions[0], state_points[0])
#u_safe_i, sg_active_i = sg.guide(actions[1], state_points[1])
u_safe_v = [sg.guide(u, x)[0][0] for u, x in zip(actions_v, unsafe_state_points_v)]
u_safe_i = [sg.guide(u, x)[0][0] for u, x in zip(actions_i, unsafe_state_points_i)]



F_v_a = feasible_set.project([2, 3])
fig, axis = plt.subplots(1, 1, figsize=(8, 4))
F_v_a.plot(axis, color='orange')
plt.plot([-v_lim, -v_lim], [-1, 1], 'r', linestyle="--", label="$v_\mathrm{lim}$")
plt.plot([v_lim, v_lim], [-1, 1], 'r', linestyle="--")
plt.plot([-v_lim, v_lim], [-1, -1], 'r', linestyle="-.", label="$m_\mathrm{lim}$")
plt.plot([-v_lim, v_lim], [1, 1], 'r', linestyle="-.")
axis.grid(True)
plt.plot(unsafe_state_points_v[0][1], actions_v[0], color='blue', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{RL}$")
plt.plot(unsafe_state_points_v[0][1], u_safe_v[0], color='green', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{SG}$")
for u_sg, u_rl, x in zip(u_safe_v, actions_v, unsafe_state_points_v):
    plt.plot(x[1], u_rl, color='blue', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{RL}$")
    plt.plot(x[1], u_sg, color='green', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{SG}$")
    plt.arrow(x[1], u_rl, 0, u_sg - u_rl -0.1*np.sign(u_sg-u_rl)  , color='red', head_length = 0.07, head_width = 0.025, length_includes_head = True)
axis.set_xlim(-0.8, 0.8)
axis.set_ylim(-1.2, 1.2)
axis.legend()
axis.set_ylabel('$m_{}$')
axis.set_xlabel('$v_{C}$')
plt.show()

F_i_a = feasible_set.project([1, 3])
fig, axis = plt.subplots(1, 1, figsize=(8, 4))
F_i_a.plot(axis, color='orange')
plt.plot(unsafe_state_points_i[0][1], actions_i[0], color='blue', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{RL}$")
plt.plot(unsafe_state_points_i[0][1], u_safe_i[0], color='green', marker='o', linewidth=2, markersize=12, label="$u_\mathrm{SG}$")
for u_sg, u_rl, x in zip(u_safe_i, actions_i, unsafe_state_points_i):
    plt.plot(x[1], u_rl, color='blue', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{RL}$")
    plt.plot(x[1], u_sg, color='green', marker='o', linewidth=2, markersize=12)#, label="$u_\mathrm{SG}$")
    plt.arrow(x[1], u_rl, 0, u_sg - u_rl -0.1*np.sign(u_sg-u_rl)  , color='red', head_length = 0.07, head_width = 0.025, length_includes_head = True)
axis.set_xlim(-0.8, 0.8)
axis.set_ylim(-1.2, 1.2)
axis.legend()
axis.set_ylabel('$m_{}$')
axis.set_xlabel('$i_{L}$')
axis.grid(True)
plt.show()

setattr(Axes3D, 'arrow3D', _arrow3D)

F_iv_a = feasible_set.project([1, 2, 3])
vertices = pypoman.compute_polytope_vertices(F_iv_a.A, F_iv_a.b)
vertices = np.asarray(vertices)
hull = ConvexHull(vertices)

hull = ConvexHull(vertices)



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
                  'figure.figsize': [5.2, 5.625],#[4.5, 7.5],
                  'font.family': 'Helvetica',#'serif',
                  'lines.linewidth': 1.2,
                  }
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Helvetica"
#})
matplotlib.rcParams.update(params)



fig = plt.figure(figsize=(5.5, 4.5))
ax = fig.add_subplot(111, projection="3d")
# draw the polygons of the convex hull
ax.scatter(unsafe_state_points_v[0][0], unsafe_state_points_v[0][1], actions_v[0],
           marker='o', s=250, color='red', alpha=1, label="$u_{\mathrm{RL}}$")
ax.scatter(unsafe_state_points_v[0][0], unsafe_state_points_v[0][1], u_safe_v[0]+0.125,
           marker='o', s=250, color='blue', alpha=1, label="$u_{\mathrm{SG}}$")
ax.arrow3D(unsafe_state_points_v[0][0],unsafe_state_points_v[0][1],actions_v[0],
           #unsafe_state_points_v[0][0],unsafe_state_points_v[0][1],u_safe_v[0],
           0, 0,u_safe_v[0]-actions_v[0]+0.3,
           mutation_scale=30,
           fc='black',
           ec ='black')
for s in hull.simplices:

    tri = Poly3DCollection([vertices[s]])
    tri.set_alpha(0.3)
    tri.set_color('green')
    ax.add_collection3d(tri)

    #tri = Line3DCollection([vertices[s]])
    #ax.add_collection3d(tri)

# draw the vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           marker='o', s=20, color='black', alpha=0.5)
ax.set_ylabel('$v_{\mathrm{}}\,/\,v_{\mathrm{lim}}$')
ax.set_xlabel('$i_{\mathrm{}}\,/\,i_{\mathrm{lim}}$')
ax.set_zlabel('$u_{\mathrm{}}$')
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('$u_{\mathrm{}}\,/\,(v_{\mathrm{DC}}/2)$', rotation=90)
ax.view_init(20, 20)
ax.legend(bbox_to_anchor = (0., 0.7, 0.25, 0.2),mode="expand", borderaxespad=0, ncol=1)
#ax.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=1)
plt.show()

folder_name = "plots_paper"
save_results = True
if save_results:
    fig.savefig(f'{folder_name}/Feasible_set_smallv.pgf')
    fig.savefig(f'{folder_name}/Feasible_set_smallv.png')
    fig.savefig(f'{folder_name}/Feasible_set_smallv.pdf')

