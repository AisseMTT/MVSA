#
# This function estimates the subspace and the afine space related with the
# observation model
#
#     y = M*x+noise
#
# where M is the mixing matrix containing the endmembers, x is the
# the fractions (sources) of each enmember at each pixel, and noise is a
# an  additive perturbation.
#

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt

def dataProj(y, p, *args): 
    # endmember matrix size
    L, samp_size = y.shape  # ((L-> number of bands, samp_sise -> sample size)
    
# ------------------------------------------------------------------------------
#  Set the defaults for the optional parameters
# ------------------------------------------------------------------------------

    proj_type = "affine"
    
# ------------------------------------------------------------------------------
#  Read the optional parameters
# ------------------------------------------------------------------------------

    if (length(*args) % 2 == 1):
        raise ValueError("Optional parameters should always go by pairs")
    else:
        for i in range(1, 2, length(*args)-1):
            if (arg[i].upper() == "PROJ_TYPE"):
                proj_type = = arg(i+1)
            else:
                # Hmmm, something wrong with the parameter string
                raise ValueError(["Unrecognized option: varargin{i}"])

# ------------------------------------------------------------------------------
#  projections
# ------------------------------------------------------------------------------

    my = y.mean(1)
    if (proj_type == "ml"):
        Up, D = svds(y * y.getH() / samp_size, p)
                                # compute the p largest singular values and the
                                # corresponding singular vectors
        sing_val = block_diag(D)
        yp = Up.getH() * y              # project onto the subspace span{E}
    elif (proj_type == "affine"):
        yp = y - np.matlib.repmat(my, 1, samp_size)
        Up, D = svds(yp * yp.getH() / samp_size, p-1)
        # represent yp in the subspace R^p
        yp = Up * Up.getH() * yp
        # lift yp
        yp = yp + np.matlib.repmat(my, 1, samp_size)
        # compute the orthogonal componeny of my
        my_ortho = my - Up * Up.getH() * my
        # define anothre orthonormal direction
        Up = Up, my_ortho/sqrt(np.sum(np.power(my_ortho,2)));
        sing_val = block_diag(D);


    return yp, Up, my, sing_val
        
    
            
    
