# DIRICHLET Random matrices from dirichlet distribution.
#
#   R = DIRICHLET_RND(A,DIM) returns a matrix of random numbers chosen   
#   from the dirichlet distribution with parameters vector A.
#   Size of R is (N x N) where N is the size of A or (N x DIM) if DIM is given.
#
#
# Modification of the dirichlet_rnd function:
# A standard Dirichlet distribution is obtained from independent gamma
# distribuitions with  scale parameters a1,...,ak, and shape set to 1
#

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt
from scipy.stats import gamma

def dirichlet(a, dim):
    rows, columbs = a.shape
    dim = rows * columns
    if np.prod(dim.shape) != 1:
        raise ValueError("The second parameter must be a scalar.")
    if rows != 1 and columns != 1:
        raise ValueError("Requires a vector as an argument.")
    # fastest method that generalize method used to sample from
    # the BETA distribuition: Draw x1,...,xk from independent gamma 
    # distribuitions with  scale and  parameters a1,...,ak, and shape set to
    # one, for each j let rj=xj/(x1+...+xk).


    N = rows * columns
    for i in range N+1:
        # generates dim random variables
        x[:, i] = gamma.rvs(a[i], 1, dim, 1)    # generates dim random variables
                                                # with gamma distribution
    r = np.divide(x, np.matlib.repmat(np.sum(x, 2), [1, N]))

    return r
        
