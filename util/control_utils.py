""" helper functions to setup MPC and/ or RL control loop """
import pypoman
import scipy
import numpy as np
import polytope as poly
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import signal
from scipy.spatial import ConvexHull
from tqdm import tqdm
from tqdm.notebook import tqdm as notetqdm
import time


def get_state_space(a, obs_limits, act_limits, tau=1, discrete=True, build_from='coeff'):
    """ set up state space matrices and discretize if necessary """
    # state space model
    if build_from == 'coeff':
        a0, a1, a2 = a
        b0 = 1/a2

        A, B = np.array([[0, 1], [-a0, -a1]]), np.array([[0], [1]])
        C, D = np.array([[b0, 0]]), np.array([[0]])

    elif build_from == 'mat':
        A, B, C, D = a

    n, m = B.shape
    # boxconstraints for states (Mx) and inputs (Mu)
    Mx, wx = np.concatenate([np.eye(n), -np.eye(n)]), np.tile(obs_limits, 2)
    Mu, wu = np.array([[1]*m, [-1]*m]), np.tile(act_limits, 2)
    Q, R = np.eye(n), np.eye(m)
    S = 2*Q

    if discrete:
        # discetize state-space model
        sys_d = signal.cont2discrete((A, B, C, D), tau, method='euler')
        A_d, B_d, C_d, D_d, _ = sys_d
        return A_d, B_d, C_d, D_d, Mx, wx, Mu, wu, Q, R, S
    else:
        return A, B, C, D, Mx, wx, Mu, wu, Q, R, S


def condensed_state_space(A, B, Q, R, S, N):
    """ build state-space block-matrices for given horizon """
    A_N, B_N = None, None
    Q_N, R_N = None, None

    n, m = B.shape

    if N == 1:
        A_N, B_N = A, B
        Q_N, R_N = Q, R
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
            B_N_row = np.squeeze(B_N_row)

            if B_N is None:
                B_N = B_N_row
            else:
                B_N = np.concatenate([B_N, B_N_row])

        Q_N = scipy.linalg.block_diag(*([Q] * N))
        Q_N = scipy.linalg.block_diag(Q_N, S)
        R_N = scipy.linalg.block_diag(*([R] * N))

    return A_N, B_N, Q_N, R_N


def condensed_constraints(Mx, wx, Mu, wu, N):
    """ build constraint block matrices for given horizon """
    if N == 1:
        Mx_N, wx_N = Mx, wx
        Mu_N, wu_N = Mu, wu
    else:
        Mx_N = scipy.linalg.block_diag(*([Mx] * (N + 1)))
        wx_N = np.concatenate([*([wx] * (N + 1))])

        Mu_N = scipy.linalg.block_diag(*([Mu] * N))
        wu_N = np.concatenate([*([wu] * N)])

    return Mx_N, wx_N, Mu_N, wu_N


def get_inequality_matrices(A_N, B_N, Mx_N, wx_N, Mu_N, wu_N, n, m,
                            x=None, qp_type='standard'):
    """ calc G, E, d for parametric QP or G, d for standart QP """
    G = np.concatenate((Mx_N @ B_N, Mu_N))
    E = np.concatenate((- Mx_N @ A_N,
                        np.zeros((wu_N.shape[0], n))))
    d = np.concatenate((wx_N, wu_N))

    if qp_type == 'standard':
        e = d + (E @ x)
        return G, e

    elif qp_type == 'parametric':
        return G, E, d


def get_polytope(G, E, d, Mx, wx, projection_dim=[1, 2]):
    """ calculate polyhedron that describes feasible set of QP """

    constr_block = np.block(
        [[Mx, np.zeros((Mx.shape[0], G.shape[-1]))],
         [-E, G]])
    constr_vec = np.block([wx, d])
    # calculate polytope and project it in the state-space
    F = poly.Polytope(constr_block, constr_vec)
    # project feasible polyhedron in given dimensions (can get computational demanding)
    F = F.project(projection_dim)

    vertices = pypoman.compute_polytope_vertices(F.A, F.b)
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
    time.sleep(1)

    return F


def calc_feasible_set2(
        A, B, Q, R, Mx, wx, Mu, wu,
        project_dim=[1, 2, 3],
        tol=0.5,
        return_all=False,
        N_max=150,
        discount=1,
        N_start=1,
        verbose=0,
        notebook_bar=False,
        progress_bar=False,
    ):
    """ calculate feasible set for increasing horizon until convergence

    args:
        A, B, Q, R, Mx, wx, Mu, wu:
            State-Space matrices and constraint inequality matrices
        project_dim (list): dimensions for the feasible-set
        tol (float): convergence tolerance
        return_all (bool): return all set during the recursive calculation
        N_max (int): Maximal prediction horizon N
        N_start (int):  Number of the amout the horizon N is increased during
                  calculations starting at 1
        discount (float): Discount factor of N_start (0, 1]
                          => increasing N gets smaller, when convergence is expected
                          N_start=1, discount=1 : every set for every N is calulated until convergence
        verbose (int): 1: prints info if set found
                       2: returns additional information
        notebook_bar (bool): Activate progress bar for jupyter notebook
        progress_bar (bool): Activate progress bar for terminal
    """
    Fs, diffs = [], []
    n, m = B.shape

    if notebook_bar:
        progress_bar = False
        pbar = notetqdm(total=N_max)
        pbar.update(0)
        pbar.set_postfix_str(f"volume difference: {(0):.5f}")
    elif progress_bar:
        notebook_bar = False
        pbar = tqdm(total=N_max)
        pbar.update(0)
        pbar.set_postfix_str(f"volume difference: {(0):.5f}")

    A_N, B_N, Q_N, R_N = condensed_state_space(A, B, Q, R, 2 * Q, 1)
    Mx_N, wx_N, Mu_N, wu_N = condensed_constraints(Mx, wx, Mu, wu, 1)
    G, E, d = get_inequality_matrices(A_N, B_N, Mx_N, wx_N, Mu_N, wu_N, n, m,
                                      qp_type='parametric')
    F_init = get_polytope(G, E, d, Mx, wx, projection_dim=project_dim)

    F_old = F_init
    Fs.append(F_old)

    N_diff = [np.ceil(N_start * discount ** (N_max - i)) for i in range(N_max)]
    N_diff = N_diff[::-1]
    N = 2

    for n_diff in N_diff:
        start_time = time.time()
        A_N, B_N, Q_N, R_N = condensed_state_space(A, B, Q, R,  2 * Q, N)
        Mx_N, wx_N, Mu_N, wu_N = condensed_constraints(Mx, wx, Mu, wu, N)
        G, E, d = get_inequality_matrices(A_N, B_N, Mx_N, wx_N, Mu_N, wu_N,
                                          n, m, qp_type='parametric')
        F_new = get_polytope(G, E, d, Mx, wx, projection_dim=project_dim)

        diff = F_old.diff(F_new)
        Fs.append(F_new)
        F_old = F_new

        end_time = time.time()

        if progress_bar or notebook_bar:
            pbar.update(n_diff)
            pbar.set_postfix_str(f"volume difference: {(diff.volume):.3f}")

        duration = (end_time - start_time) / 60
        if duration >= 5.0:
            print('returned due to long execution time')
            if verbose == 1:
                print('\nFeasible-Set found\n')

            if return_all:
                if verbose == 2:
                    return Fs, diffs
                else:
                    return Fs
            else:
                if verbose == 2:
                    return Fs[-1], diffs
                else:
                    return Fs[-1]

        if diff.volume <= tol:
            if verbose == 1:
                print('\nFeasible-Set found\n')
            elif verbose == 2:
                pass

            if return_all:
                if verbose == 2:
                    return Fs, diffs
                else:
                    return Fs
            else:
                if verbose == 2:
                    return Fs[-1], diffs
                else:
                    return Fs[-1]

        N += n_diff
        N = int(N)
        diffs.append(diff.volume)

    print(f'not converged after a horizon of {N_max}')