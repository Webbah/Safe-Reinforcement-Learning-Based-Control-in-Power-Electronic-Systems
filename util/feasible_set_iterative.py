from datetime import time

import numpy as np
import scipy
import polytope as poly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pypoman
from scipy.spatial import ConvexHull



def plot_polytope_3d(ploy):
    vertices = pypoman.compute_polytope_vertices(ploy.A, ploy.b)
    vertices = np.asarray(vertices)
    hull = ConvexHull(vertices)

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

def plot_polytope_2d(poly, wx):
    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    poly.plot(axis, color='orange')
    axis.set_xlim(-wx[0] - 0.2, wx[0] + 0.2)
    axis.set_ylim(-wx[2] - 0.2, wx[2] + 0.2)
    axis.set_xlabel('$i_{L}$')
    axis.set_ylabel('$v_{C}$')
    axis.grid(True)
    #plt.show()

def get_polytope(G, E, d, Wx, omega_x, projection_dim=[1, 2], plt_all=False):
    """ calculate polyhedron that describes feasible set of QP """

    constr_block = np.block(
        [[Wx, np.zeros((Wx.shape[0], G.shape[-1]))],
         [-E, G]])
    constr_vec = np.block([omega_x, d])
    # calculate polytope and project it in the state-space
    F = poly.Polytope(constr_block, constr_vec)
    # print(F)

    #vertices = pypoman.compute_polytope_vertices(F.A, F.b)


    # project feasible polyhedron in given dimensions (can get computational demanding)
    F = F.project(projection_dim)
    # print(F.volume)
    if plt_all:
        if len(projection_dim) == 3:
            plot_polytope_3d(F)
        elif len(projection_dim) == 2:
            plot_polytope_2d(F, omega_x)

        time.sleep(1)

    return F

def condensed_constraints(Wx, omega_x, Wu, omega_u, N):
    """ build constraint block matrices for given horizon """
    if N == 0:
        Wx_N, omega_x_N = Wx, omega_x
        Wu_N, omega_u_N = Wu, omega_u
    else:
        Wx_N = scipy.linalg.block_diag(*([Wx] * (N + 1)))
        omega_x_N = np.concatenate([*([omega_x] * (N + 1))])

        Wu_N = scipy.linalg.block_diag(*([Wu] * N))
        omega_u_N = np.concatenate([*([omega_u] * N)])

    return Wx_N, omega_x_N, Wu_N, omega_u_N

def get_inequality_matrices(A_N, B_N, Wx_N, omega_x_N, Wu_N, omega_u_N, n, m,
                            x=None, qp_type='standard'):
    """ calc G, E, d for parametric QP or G, d for standart QP """
    G = np.concatenate((Wx_N @ B_N, Wu_N))
    E = np.concatenate((- Wx_N @ A_N,
                        np.zeros((omega_u_N.shape[0], n))))
    d = np.concatenate((omega_x_N, omega_u_N))

    if qp_type == 'standard':
        e = d + (E @ x)
        return G, e

    elif qp_type == 'parametric':
        return G, E, d

def condensed_state_space(A, B, N):
    """ build state-space block-matrices for given horizon """
    A_N, B_N = None, None
    n, m = B.shape

    if N == 0:
        #A_N, B_N = A, B
        A_N, B_N = np.eye(2), np.zeros([2, 1])

    else:
        for i in np.arange(0, N + 1):
            # build columns
            if A_N is None:
                A_N = np.linalg.matrix_power(A, i)
            else:
                A_N = np.concatenate(
                    [A_N, np.linalg.matrix_power(A, i)], axis=0
                )

            B_N_row = None
            for j in range(N):
                # build rows
                power = i - j - 1
                if power < 0:
                    if B_N_row is None:
                        B_N_row = np.zeros((n, m))
                    else:
                        B_N_row = np.concatenate(
                            [B_N_row, np.zeros((n, m))], axis=-1
                        )
                else:
                    if B_N_row is None:
                        B_N_row = np.linalg.matrix_power(A, power) @ B
                    else:
                        B_N_row = np.concatenate(
                            [B_N_row, np.linalg.matrix_power(A, power) @ B],
                            axis=-1,
                        )
            #B_N_row = np.squeeze(B_N_row)

            if B_N is None:
                B_N = B_N_row
            else:
                B_N = np.concatenate([B_N, B_N_row])



    return A_N, B_N

def calc_feasible_set(A, B, Wx, omega_x, Wu, omega_u, project_dim, return_all, N_max, N_start=1):
    Fs, diffs = [], []
    n, m = B.shape

    Wx_N, omega_x_N, Wu_N, omega_u_N = condensed_constraints(Wx, omega_x, Wu, omega_u, 0)
    G, E, d = get_inequality_matrices(np.eye(n), np.zeros([n, m]), Wx_N, omega_x_N, Wu_N, omega_u_N, n, m,
                                      qp_type='parametric')



    F_init = get_polytope(G, E, d, Wx, omega_x, projection_dim=project_dim, plt_all=False)

    Fs.append(F_init)
    F_old = F_init

    #F = F_init.project([1, 2])
    #plot_polytope_2d(F, omega_x)
    #plt.title(f"N = 0")
    #plt.show()
    #print(f"2D volume: {F.volume} bei N = 0")

    N_diff = [np.ceil(N_start *  (N_max - i)) for i in range(N_max)]
    N_diff = N_diff[::-1]
    N = 1

    for n_diff in range(N_max):
        A_N, B_N = condensed_state_space(A, B, N)
        Wx_N, omega_x_N, Wu_N, omega_u_N = condensed_constraints(Wx, omega_x, Wu, omega_u, N)
        G, E, d = get_inequality_matrices(A_N, B_N, Wx_N, omega_x_N, Wu_N, omega_u_N,
                                          n, m, qp_type='parametric')
        F_new = get_polytope(G, E, d, Wx, omega_x, projection_dim=project_dim, plt_all=False)

        diff = F_old.diff(F_new)
        Fs.append(F_new)
        F_old = F_new

        F = F_new.project([1, 2])
        #plot_polytope_2d(F, omega_x)
        #plt.title(f"N = {N}")
        #plt.show()
        #print(f"2D volume: {F.volume} bei N = {N}")
        # print(f"3D polytope diff: {diff} bei N = {N}")
        #diffs.append(diff.volume)

        N += 1
        N = int(N)

    return F_new.project(project_dim)

