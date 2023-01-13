
from collections import deque
import numpy as np
import padasip as sip



class RLSFit:
    """ class utilizing existing RLS filter, adding a few helping features """

    def __init__(self, state_dim, action_dim, mu=0.5, buffer_len=1):
        """
        args:

        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # setup measurement vector
        self.xi = None
        self.next_state = None

        self.estimator = [
            sip.filters.FilterRLS(n=state_dim+action_dim, mu=mu) for i in range(state_dim)
        ]

        self.error_buffer = deque(maxlen=buffer_len)

    def fit(self, state_act, next_state, return_mode=None):
        self.xi = np.concatenate(state_act)
        self.next_state = next_state

        for i in range(self.state_dim):
            self.estimator[i].adapt(next_state[i], self.xi)

        state_coeffs = [est.w[:self.state_dim][None, :] for est in self.estimator]
        action_coeffs = [est.w[self.state_dim:][None, :] for est in self.estimator]
        A_hat = np.concatenate(state_coeffs, axis=0)
        B_hat = np.concatenate(action_coeffs, axis=0)

        if return_mode == 'all':
            return (state_coeffs, action_coeffs), (A_hat, B_hat)
        elif return_mode == 'coeff':
            return (state_coeffs, action_coeffs)
        else:
            return

    def predict(self, state_act=None):
        if state_act is not None:
            xi = np.concatenate(state_act)
        else:
            xi = self.xi

        predictions = np.asarray([est.predict(xi) for est in self.estimator])
        return predictions

    def calc_error(self, metric='MSE'):
        if metric == 'AE':
            # calc absolute prediction error for each state
            predictions = self.predict()
            abs_error = np.abs(self.next_state - predictions)
            return abs_error

        elif metric == 'Error':
            # calc direct error
            predictions = self.predict()
            error = self.next_state - predictions
            return error

        elif metric == 'MSE':
            # TODO add other error metrics
            # calc mean squared error over error-buffer
            raise NotImplementedError

        else:
            raise NotImplementedError





