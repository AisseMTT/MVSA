
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt

def hinge(y):
#  z = hinge(y)
#
#   hinge function)

z = np.amax(-y, 0);

return z 
