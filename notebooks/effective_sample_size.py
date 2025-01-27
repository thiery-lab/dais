import numpy as np


def ESS(log_w_unormalized):
    """ compute the effective sample size of a set of weights """
    log_w = log_w_unormalized - np.max(log_w_unormalized)
    w = np.exp(log_w)
    w = w / np.sum(w)
    return 1./(np.sum(w**2) * len(w))


def next_temperature(
                log_w_init,  # (S,):       initial log weights
                ess_target,  # float:      target ESS
                tol=10**-5,  # float:      tolerance for the bissection
                ):
    """
    compute the next temperature for a target effective sample size
    using a bissection algorithm
    """

    temp_min = 0
    temp_max = 1

    if ESS(temp_max*log_w_init) >= ess_target:
        return temp_max

    while temp_max - temp_min > tol:
        temp_mid = (temp_min + temp_max)/2.
        ESS_new = ESS(temp_mid*log_w_init)

        if ESS_new < ess_target:
            temp_max = temp_mid
        else:
            temp_min = temp_mid
    return (temp_min + temp_max)/2.
