import numpy as np
import scipy.linalg


def is_psd(matrix):
    """
    checks if a matrix is positive semi-definite
    by attempting to compute the cholesky decomposition
    """
    try:
        scipy.linalg.cholesky(matrix)
        return True
    except scipy.linalg.LinAlgError:
        return False


def log_det_psd(matrix):
    """
    computes the log determinant of a positive semi-definite matrix
    by computing the cholesky decomposition and summing 
    the log of the diagonal elements
    """
    L = scipy.linalg.cholesky(matrix, lower=True)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return log_det


def invert_psd_matrix(A,            # (dim,dim): psd matrix
                      chol=np.nan,  # (dim,dim): cholesky decomposition of A
                      ):
    # make sure chol is lower triangular if given
    if not np.all(np.isnan(chol)):
        assert np.allclose(chol, np.tril(chol)), "chol is not lower triangular"

    # Perform Cholesky decomposition of A
    chol = np.where(np.isnan(chol), scipy.linalg.cholesky(A, lower=True), chol)

    # Invert L
    chol_inv = scipy.linalg.solve_triangular(
                                    chol,
                                    np.eye(chol.shape[0]),
                                    lower=True)

    # Compute the inverse of A using L_inv
    A_inv = chol_inv.T @ chol_inv
    return A_inv
