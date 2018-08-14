#
# estNoise : hyperspectral noise estimation.
# This function infers the noise in a 
# hyperspectral data set, by assuming that the 
# reflectance at a given band is well modelled 
# by a linear regression on the remaining bands.
#

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt

def estNoise(r, *args):
    # if ~isnumeric(y), error('the data set must an L x N matrix');
    noise_type = "additive"     # default value
    verbose = 1                 # default value
    verb = "on"
    for i in range(length(*args) - 1):
        if (arg[i].lower() == "additive"):
            noise_type = "additive"
        elif (arg[i].lower() == "poisson"):
            noise_type = "poisson"
        elif (arg[i].lower() == "on"):
            verbose = 1
            verb = "on"
        elif (arg[i].lower() == "off"):
            verbose = 0
            verb = "off"
        else:
            raise ValueError("parameter [%d] is unknown "+ i)

    L, N = y.shape
    if L < 2:
        raise ValueError("Too few bands to estimate the noise. ")
    
    if verbose:
        print(1,"Noise estimates:\n")

    if noise_type.equals('noise_type'):
        sqy = sqrt(np.multiply(y, (y>0)))       # prevent negative values
        u, Ru = estAdditiveNoise(sqy, verb)     # noise estimates
        x = np.power((sqy - u), 2)              # signal estimates
        w = np.multiply(sqrt(x), u) * 2
        Rw = w * w.getH() / N
    else                                        # additive noise
       w, Rw = estAdditiveNoise(y, verb)        # noise estimates

    return w, Rw

# ------------------------------------------------------------------------------
#  Internal Function 
# ------------------------------------------------------------------------------

    def estAdditiveNoise(r, verbose):
        small = 1e-6
        verbose = not verbose.lower() == 'off'
        L, N = r.shape
        # the noise estimation algorithm
        w = np.zeros(L, N)
        if verbose:
            print(1,"computing the sample correlation matrix and its inverse\n")
        RR = r * r.getH()     # equation (11)
        RRi = inv(RR + small * identity(L))     # equation (11)
        if verbose:
            print(1,"computing band    ")
        for i in range(L + 1):
            if verbose:
                print(1,"\b\b\b%3d",i)
            # equation (14) -- may need revision 
            XX = RRi - (RRi[:, i]* RRi[i, :])/RRi[i,i]
            RRa = RR[:, i]
            RRa[i] = 0      # this remove the effects of XX(:,i)
            # equation (9)
            beta = XX * RRa
            beta[i] = 0     # this remove the effects of XX(i,:)
            # equation (10)
            w[i, :] = r[i, :] - beta.getH() * r; # note that beta(i)=0 =>
                                                 # beta(i)*r(i,:)=0
            if verbose:
                print(1,"\ncomputing noise correlation matrix\n")
            Rw = block_diag(block_diag(w * w.getH() / N))

        return w, Rw
    
