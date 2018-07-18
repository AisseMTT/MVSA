#
#
# HySime: Hyperspectral signal subspace estimation
# Input:
#        y  hyperspectral data set (each column is a pixel)
#           with (L x N), where L is the number of bands
#           and N the number of pixels
#        w  (L x N) matrix with the noise in each pixel
#        Rw noise correlation matrix (L x L)
#        verbose [optional] (on)/off
# Output
#        kf signal subspace dimension
#        Ek matrix which columns are the eigenvectors that span 
#           the signal subspace
#

# import needed packages and classes
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt
import matplotlib.pyplot as plt

def hysime(y, n, Rn, verbose = true):

    y = arg[0];     # 1st parameter is the data set
    L, N = y.shape
    if not np.prod(y.shape):
        raise ValueError("the data set is empty")
    n = arg[1]      # the 2nd parameter is the noise
    Ln, Nn = n.shape
    Rn = arg[2]         # the 3rd parameter is the noise correlation matrix
    d1, d2 = Rn.shape
    #if nargin == 4, verbose = ~strcmp(lower(varargin{4}),'off');

    if ln != L or Nn != N:       # n is an empty matrix or with different size
        raise ValueError("empty noise matrix or its size does",
                         "not agree with size of y\n")
    if d1 != d2 or d1 != L:
        print("Bad noise correlation matrix\n")
        Rn = n * n.getH() / N


    x = y - n

    if verbose:
        print(1, "Computing the correlation matrices\n")
    L, N = y.shape
    Ry = y * y.getH() / N       # sample correlation matrix
    Rx = x * x.getH() / N       # signal correlation matrix estimates
    if verbose:
        print(1, "Computing the eigen vectors of the signal correlation matrix\n")
    E, D = svds(Rx)     # eigen values of Rx in decreasing order, equation (15)
    dx = block_diag(D)

    if verbose:
        print(1, "Estimating the number of endmembers\n")
    Rn = Rn + np.sum(block_diag(Rx)) / L / 10**10 * np.identity(L)
    Py = block_diag(E.getH() * Ry * E)      # equation (23)

    Pn = block_diag(E.getH() * Rn * E)      # equation (24)
    cost_F = -Py + 2 * Pn                   # equation (22)

    # syntax might need revision 
    kf = np.sum(cost_F < 0)
    dummy, ind_asc = np.sort(cost_F)
    Ek = E[:, ind_asc[1:kf]]
    if verbose:
        print(1, "The signal subspace dimension is: k = %d\n", kf)

    # only for plot purposes, equation (19)
    Py_sort =  np.trace(Ry) - np.cumsum(Py(ind_asc))
    Pn_sort = 2 * np.cumsum(Pn(ind_asc))
    cost_F_sort = Py_sort + Pn_sort

    fig = plt.figure()

    semilogy(indice, cost_F_sort(indice), indice,Py_sort(indice), indice,Pn_sort(indice), 2, 5, **kwargs)
    plt.semilogx([1, 10, 100], [1, 10, 100])
    plt.xlabel("k")
    plt.ylabel("mse(k)")
    plt.title('HySime')
    legend('Mean Squared Error','Projection Error','Noise Power')
    plt.show()

return kf, Ek 

# ------------------------------------------------------------------------------
# end of function [varargout]=hysime(varargin
# ------------------------------------------------------------------------------

    
    
        
    
        
        

    
        
    
