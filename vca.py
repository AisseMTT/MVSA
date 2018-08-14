#
#
# Vertex Component Analysis
#
# Minimum Volume Simplex Analysis: A fast Algorithm to Unmix Hyperspectral Data
# This code has been translated from MATLAB into Python 
# This file contains only the VCA Algorithm 
#

# import needed packages and classes
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
from math import exp, expm1

def VCA(R, *args):

# ------------------------------------------------------------------------------
# Default Parameters 
# ------------------------------------------------------------------------------
    verbose = 'on'  # default 
    snr_input = 0   # default this flag is 0
                    # which means we estimate the SNR
    
# ------------------------------------------------------------------------------
# Looking for input parameters
# ------------------------------------------------------------------------------    

    dim_in_par = length(*args)
    if (nargin - dim_in_par) != 1:
        raise ValueError("Wrong parameters")
    elif dim_in_par % 2 == 1:
        raise ValueError("Optional parameters should always go by pairs")
    else:
        for i in range(1, 2, dim_in_par-1):
            if(arg[i].lower() == "verbose"):
                verbose = arg(i+1)
            elif(arg[i].lower() == "endmembers"):
                p = arg(i+1)
            elif(arg[i].lower() == "snr"):
                SNR = arg(i+1)
                snr_input = 1;       # flag meaning that user gives SNR
            else:
                print(1, "Unrecognized parameter:%s\n", arg[i])

# ------------------------------------------------------------------------------
# Initializations
# ------------------------------------------------------------------------------

    if R.size:
       raise ValueError("there is no data")
    else:
        data = ([[1, 2], [3, 4], [5, 6]])   # L number of bands (channels) 
        L, N = len(data), len(data[0])      # N number of pixels (LxC)

    if p<0 or p>L or p%1 != 0:
        raise ValueError("ENDMEMBER parameter must be integer between 1 and L")

# ------------------------------------------------------------------------------
# SNR Estimates
# ------------------------------------------------------------------------------
    
    if snr_input == 0:
        r_m = R.mean(1)
        # may need to check this 
        R_m = np.tile(r_m,(1,N))    # mean of each band
        R_o = R - R_m               # data with zero-mean
        Ud, Sd, Vd = svds(R_o * R_o.getH()/N, p)    #computes the p-projection matrix
        x_p =  Ud.getH() * R_o                      # project the zero-mean data onto
                                                    # p-subspace
        SNR = estimate_snr(R,r_m,x_p)       # from another set of code
        if verbose.equals('on'):
            print(1,"SNR estimated = %g[dB]\n",SNR)
            break
    else:
        if verbose.equals('on'):
            print(1,'input    SNR = %g[dB]\t',SNR)
            break

    SNR_th = 15 + 10 * math.log10(p)

# ------------------------------------------------------------------------------
# Choosing Projective Projection or 
#         projection to p-1 subspace
# ------------------------------------------------------------------------------
            
    if SNR < SNR_th:
        if verbose.equals('on'):
            print(1,"... Select the projective proj.\n",SNR)
            d = p - 1
            break
        if snr_input == 0:      # it means that the projection is already computed
            Ud = Ud[:, 0:d]
        else:
            r_m = R.mean(1)
            R_m = np.tile(r_m, (1,N))   # mean of each band
            R_o = R - R_m;           # data with zero-mean
            Ud, Sd, Vd = svds(R_o * R_o.getH()/N, d)    # computes the p-projection
                                                        # matrix
            x_p =  Ud.getH() * R_o      # project thezeros mean data onto p-subspace
            break
        Rp = Ud * x_p[0:d, :] + np.tile(r_m, (1,N))    # again in dimension L
        x = x_p[1:d,:]      #  x_p =  Ud' * R_o; is on a p-dim subspace
        c = np.amax(np.sum(np.power(x, 2), 1))**0.5
        y = [x, c * np.ones(1,N)]
    else:
        if verbose.equals('on'):
            print(1,"... Select proj. to p-1\n",SNR)
        d = p
        Ud, Sd, Vd = svds(R * R.getH()/N, d)    # computes the p-projection matrix

        x_p = Ud.getH() * R
        Rp = Ud * x_p[0:d, :]   # again in dimension L (note that x_p has no null mean)

        x = Ud.getH() * R
        u = x.mean(1)   # equivalent to  u = Ud' * r_m
        # syntax might be an issue 
        y = np.divide(x, np.matlib.repmat(np.sum(np.multiply(x, np.matlib.repmat(u, (1,N))))), (d, 1))

# ------------------------------------------------------------------------------
# VCA algorithm
# ------------------------------------------------------------------------------
    indice = np.zeros(0, p)
    A = np.zeros(p, p)
    A[p, 0] = 1

    for i in range (p + 1):
        w = np.random.uniform(p, 1)
        f = w - A * np.linalg.pinv(A) * w
        f = f/(sqrt(np.sum(np.power(f, 2))))
        f = f / sqrt(sum(f.^2))

        v = f.getH() * y
        v_max, indice(i) = np.amax(np.absolute(v))
        A[:, i] = y[:, indice(i)]        # same as x(:,indice(i))

    Ae = Rp[:, indice)

    return Ae, indice, Rp

# ------------------------------------------------------------------------------
# Internal functions
# ------------------------------------------------------------------------------

    def estimate_snr(R, r_m, x):
        L, N = R.shape  # L number of bands (channels)
                        # N number of pixels (Lines x Columns)
        p, N = x.shape  # p number of endmembers (reduced dimension)
        P_y = np.sum(np.power(R(:), 2))/N
        P_x = np.sum(np.power(x(:), 2))/N + r_m.getH() * r_m
        snr_est = 10 * math.log10((P_x - p/L*P_y)/(P_y - P_x))

    return snr_est
        
        

        

        
        
        
        
        
        
        
            
            
        
            
        
               
    
