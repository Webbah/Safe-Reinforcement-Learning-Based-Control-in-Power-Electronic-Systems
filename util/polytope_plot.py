import pypoman
import pathlib
import sys
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

dir_list = str(pathlib.Path().resolve()).split("/")
parent_idx = np.where(np.array(dir_list) == "masc")[0][0]
parent_path = pathlib.Path().resolve()
for idx in range(len(dir_list) - parent_idx - 1):
    parent_path = parent_path.parent
sys.path.append(str(parent_path) + "/")

import feasible_helper


# state space parameters
a = (0, 0, 1)
obs_limits, act_limits = np.array([25, 5]), np.array([1])
init_limits = np.array([15, 3])
tau = 1
A_d, B_d, C_d, D_d, Mx, wx, Mu, wu, Q, R, S = feasible_helper.get_state_space(
    a, obs_limits, act_limits, tau
)
n, m = B_d.shape

feasible_set3d = feasible_helper.calc_feasible_set(
    A_d, B_d, Q, R, Mx, wx, Mu, wu, project_dim=[1, 2, 3], N_max=50, tol=0.01,
    discount=1, N_start=1, progress_bar=True)

feasible_set2d = feasible_helper.calc_feasible_set(
    A_d, B_d, Q, R, Mx, wx, Mu, wu, project_dim=[1, 2], N_max=50, tol=0.01,
    discount=1, N_start=1, progress_bar=True)

vertices = pypoman.compute_polytope_vertices(feasible_set3d.A, feasible_set3d.b)
vertices = np.asarray(vertices)
hull = ConvexHull(vertices)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection="3d")
# draw the polygons of the convex hull
for s in hull.simplices:
    tri = Poly3DCollection([vertices[s]])
    tri.set_alpha(0.5)
    tri.set_color('dodgerblue')
    ax.add_collection3d(tri)

# draw the vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
           marker='o', s=10, color='orange', alpha=0.5)
plt.show()

fig, axis = plt.subplots(1, 1, figsize=(8, 4))
feasible_set2d.plot(axis, color='orange')
axis.set_xlim(-wx[0]-1, wx[0]+1)
axis.set_ylim(-wx[1]-0.2, wx[1]+0.2)
axis.set_xlabel('$x_{1}$')
axis.set_ylabel('$x_{2}$')
axis.grid(True)
plt.show()