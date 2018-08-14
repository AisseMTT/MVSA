
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt

def soft_neg(y, tau):
    #  z = soft_neg(y,tau);
    #
    #  negative soft (proximal operator of the hinge function)

    z = np.amax(np.absolute(y + tau / 2) - tau / 2, 0);
    z = np.divide(z, np.multiply(z + tau / 2,  y + tau / 2));
    
    return z 
