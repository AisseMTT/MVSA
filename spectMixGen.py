import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import block_diag
from numpy.linalg import inv
import warnings
import sys
import cvxopt
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
import scipy.io as sio
import math

def function spectMixGen(M,samp_size,varargin):

# Usage:
# [y,x,noise,outliers] = spectMixGen(M,samp_size,varargin)
#
# This function generates a simulated spectral data set
#
#     y = M*x+noise
#
# where M is the mixing matrix containing the endmembers, x is the
# the fractions (sources) of each enmember at each pixel, and noise is a
# Gaussian independent (band and pixel - wise) additive perturbation.
# 
#
# 
# Author: Jose Bioucas-Dias, January, 2009.
# 
# Please check for the latest version of the code and papers at
# www.lx.it.pt/~bioucas/SUNMIX
#
# 
#
#
#  =====================================================================
#  ===== Required inputs ===============================================
#
#  M: [Lxp] mixing matrix containing the p endmembers of size L 
#     
#  samp_size: number of spectral vectors to be generated
#
#  
#  ===== Optional inputs =============
# 
#  
#  'Source_pdf' = Source densities  {'uniform', 'Diri_id', 'Diri_mix'}
#                'uniform'   -> uniform over the simplex
#                'Diri_id'   -> Direchlet with equal parameters
#                'Diri_mix'  -> Mixture of Direchlet densities
#
#                Default = 'uniform'
#
#  'pdf_pars'  = pdf parameters 
#                if ('Source_pdf' == 'Diri_id') 
#                      pdf_pars = a > 0 ; p(x) \sim D(a,...a) 
#                if ('Source_pdf' == 'Diri_mix')
#                      pdf_pars = A; [m,p+1] ;
#                      Each line of A conatins the parameters of a 
#                      Direchlet mode. 
#                      A(i,1)    -> weights of mode i  (0 < A(i,1)<= 1, sum(A(:,1)) = 1)
#                      A(i,2:p)  -> Dirichelet parameters of mode i  
#                                   (A(i,j)> 0, j>=2)
#                Default = 1 (<==> uniform);
#
#   'pure_pixels'  = include pure pixels in the data set  {'yes', 'no'}
#                    
#                   Default = 'no' 
#
#    'max_purity' = vector containing maximum purities. Is a scalat is
#                   passed, it is interpreted as a vector with components
#                   equal to the passed vscalar.
#
#                 Default = [1 , ..., 1]
#
#    'am_modulation' = multiplies the sources by a random scale factor
#                      uniformely distributed in an the interval.
#
#                   Default = [1 1]  % <==> no amplitude modulation
#
#    'sig_variability' = multiply each component of each source with a 
#                        a random scale factor uniformely distributed in an the interval
#
#                   Default = [1 1]  % <==> no asignature variability.
#                           
#
#    'no_outliers' = number of vector outside the simplex (sum(x) = 1, but some x_i > 1 )
#
#                    Default = 0;
#
#    'violation_extremes' = [min(x), max(x)] in case of  no_outliers > 0
#
#                           Default = [1 1.2];
#    
#    'snr' = signal-to-noise ratio in dBs    
# 
#           Default = 40 dBs
#                   
#
#    'noise_shape' = shape of the Gaussian variace along the bands
#                    {'uniform', 'gaussian', 'step', 'rectangular'}
#                     'uniform'     -> equal variance
#                     'gaussian'    -> Gaussian shaped variance centered at
#                                      b and with spread eta:
#                                      1+ beta*exp(-[(i-b)/eta]^2/2)
#                     'step'        -> step  and amplitudes (1,beta) centered
#                                      at b 
#                     'rectangular' ->  1+ beta*rect(i-b)/eta
# 
#                     Default = 'uniform';
#
#    'noise_pars'  = noise parameters [beta  b eta]
#
#                    Default = [0 1 1];  <==> 'uniform'
#
#    
#
# ===================================================  
# ============ Outputs ==============================
#
# [y,x,noise,outliers]
#   y = [Lxsamp_size] data set
#
#   x =  [pxsamp_size] fractional abundances
#   
#   noise = [Lxsamp_size] additive noise 
#
#   sigma = [Lx1]  vector with the noise standard deviations along the L bands
#
#   outliers = [pxno_outliers] source oultliers
#
# ========================================================
#
# NOTE: The order in which the degradation mechanisms are input is
# irrelevant. However, since the degradation mechanism are not comutative,
# the  implemented order is relevent and it is the following:
#
#    1)  generate sources 
#    2)  enforces max purity
#    3)  include pure pixels 
#    4)  include outliers 
#    5)  amplitude modulation 
#    6)  signatute variability
#    7)  generate noise
#    
# ===================================================  
# ============ Call examples ==============================
#
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size)
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_id', 'pdf_pars', 5)
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_mix', 'pdf_pars', [1, 0.1 1 2 3])
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', ...
#                                                    'Diri_mix', ...'pdf_pars', [0.2, 0.1 1 2 3,
#                                                                                0.8, 2   3 4 5])
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size, 'Source_pdf', 'Diri_id',
#                     'snr', 20, 'pdf_pars', 5,  'max_purity', [0.8 1 1 1],  'noise_shape',
#                      'gaussian', 'noise_pars', [2,50,20])
#
#  [y,x,noise,outliers] = spectMixGen(M,samp_size)
#







#--------------------------------------------------------------
# test for number of required parametres
#--------------------------------------------------------------

    # endmember matrix size 
    L, p = M.shape  #((L-> number of bands, p -> number of endmembers)
    

#--------------------------------------------------------------
# Set the defaults for the optional parameters
#--------------------------------------------------------------

    source_pdf   = 'uniform'
    pdf_pars = 1
    pure_pixels = 'no'
    max_purity = np.ones(1, p)
    no_outliers = 0
    violation_extremes = [1, 1.2]
    am_modulation = [1, 1]
    sig_variability = [1, 1]
    snr = 40     # 40 dBs
    noise_shape = 'uniform'
    noise_pars = [0, 1, 1]


#--------------------------------------------------------------
# Read the optional parameters
#--------------------------------------------------------------

    if (length(*args) % 2 == 1):
        error('Optional parameters should always go by pairs')
    else:
        for i in range(1, 2, length(*args)-1):
            if (arg[i].upper() == "SOURCE_PDF"):
                source_pdf = arg(i+1)
            elif (arg[i].upper() == "PDF_PARS"):
                pdf_pars = arg(i+1)
            elif (arg[i].upper() == "PURE_PIXELS"):
                pure_pixels = arg(i+1)
            elif (arg[i].upper() == "MAX_PURITY"):
                max_purity = arg(i+1)
            elif (arg[i].upper() == "NO_OUTLIERS"):
                no_outliers = arg(i+1)
            elif (arg[i].upper() == "VIOLATION_EXTREMES"):
                violation_extremes = arg(i+1)
            elif (arg[i].upper() == "AM_MODULATION"):
                am_modulation = arg(i+1)
            elif (arg[i].upper() == "SIG_VARIABILITY"):
                sig_variability = arg(i+1)
            elif (arg[i].upper() == "SNR"):
                snr = arg(i+1)
            elif (arg[i].upper() == "NOISE_SHAPE"):
                noise_shape = arg(i+1)
            elif (arg[i].upper() == "NOISE_PARS"):
                noise_pars = arg(i+1)
            else:
                raise ValueError("Unrecognized option:" arg(i))

## generate sources 
# check for validity

    if source_pdf.equals('Diri_mix'):
        no_modes, cols = pdf_pars.shape
        if cols is not (p+1):
            raise ValueError("Wrong pdf parameters")
        elif (np.sum(pdf_pars[:,1] != 1) || (np.sum(pdf_pars[:,1] < 0) > 0):
            raise ValueError("Wrong pdf parameters -> mixing weights  do not define a probability")
    else:
        no_modes = 1       
    
    # take the density as a mixture in all cases (weights - MOD weights; pdf_pars(i,:) - MOD_i parameters')
    if (source_pdf == 'uniform'):
        pdf_pars = np.ones(1,p)
        weights = 1
    elif (source_pdf == 'Diri_id')
        pdf_pars = pdf_pars[1]*np.ones(1,p)
        weights = 1
    elif (source_pdf == 'Diri_mix')
        weights = pdf_pars[:,1]
        pdf_pars = pdf_pars[:, 2:-1]  
     
     # determine  the size of each lenght
     mode_length = int(round(weights*samp_size))
 
     # correct for rounding erros
     mode_length[no_modes] = mode_length[no_modes] + samp_size - np.sum(mode_length)
 
     x = [ ]
     for i in range(0, no_modes + 1):
        x = [x, dirichlet(pdf_pars[i,:],mode_length[i]).conj().T]
 
     # do a random permutation of columns (not necessary)
     x = x[:, np.random.permutation(samp_size)]
 

    ## enforces max purity

    # if max_purity is a scalar, convert it into a vector
    if length(max_purity) == 1:
        max_purity = max_purity * np.ones(1,p)


    # check for validity
    if (np.sum(max_purity < 0 ) + np.sum(max_purity > 1)) > 0:
        raise ValueError("Purity must be in (0,1)")            
    elif np.sum(max_purity) <1
        raise ValueError("Purity must be in (0,1)")
    
    # ensure that is a line vector
    max_purity = np.reshape(max_purity,1,p)

    if np.sum(max_purity) < p
        # threshold sources
        x_max = np.matlib.repmat(max_purity.getH(),1,samp_size)
        # build a  mask
        mask = (x <= x_max)
        # threshold sources
        x_th = np.multiply(x, mask) + np.multiply((1-mask), x_max)
        # sources excess
        x_excess = x-x_th
        # total of excess per pixel
        x_acum = np.sum(x_excess)
        # slack per peixel
        slack = x_max - x_th
        # redistribute total acumulated proportionali to the individual slack
        x = np.divide(x_th+slack, np.multiply(np.matlib.repmat(np.sum(slack), p, 1),np.matlib.repmat(x_acum, p, 1))) 

    ## include pure pixels (at the end)
    if pure_pixels.equals('yes'):
        x = [x[:,p+1:-1], np.identity(p)]
 
    ## include outliers (at the begining)
    # we simple pick up the first no_outliers pixels and forces one of its 
    # fractions  to be in the interval violation_extremes an to sum 1
    spread = violation_extremes[2]-violation_extremes[1]
    for i in range(0, no_outliers+1):
        index = np.random.permutation(p)
        x[index[1],i] = violation_extremes[1] + spread*np.random.uniform(1)
        aux = np.random.uniform(p-1,1)
        aux = np.divide(aux, np.matlib.repmat(np.sum(aux), p-1, 1)) - x[index[1],i]/(p-1)
        x[index[2:p],i] = aux

    outliers = x[:,1:no_outliers]

    ## amplitude modulation 
    am_length = am_modulation[2] - am_modulation[1]
    if ((am_length) > 0 ) and ((am_modulation[1]) >= 0): # apply amplitude modulation only if the 
                                                        # interval in positive
        x = np.multiply(x, np.matlib.repmat(am_modulation[1] + am_length*np.random.uniform(1, x.shape(2)), p, 1))

    ## signature  variability 
    sig_length = sig_variability[2]- sig_variability[1]
    if ((sig_length) > 0 ) and (sig_variability(1) >= 0):   # apply amplitude modulation only if the 
                                                            # the interval in positive
        x = np.multiply(sig_variability[1] + sig_length * np.random.uniform(x.shape))

    ## generate noise
 
    # generate noise shape (sum of variances is one)
    beta = noise_pars[1] 
    b = noise_pars[2] 
    eta = noise_pars[3] 

    xx = [1:L].conj().T

    if noise_shape.equals('uniform'):    
        sigma = np.ones(L,1)
    elif noise_shape.equals('gaussian'):
        sigma = 1 + beta*math.exp(np.power(-[(xx-b)/eta], 2)/2)
    elif noise_shape.equals('step'):
        sigma = 1 + beta * (xx >= b)
    elif noise_shape.equals('rectangular'):
        sigma = 1 + beta*(np.absolute((xx-b)/eta) < 1)
 


    # normalize
    sigma = sigma/sqrt(np.sum(np.power(sigma,2)))
 
    # compute mean variance
    # generate data without noise
    y=M*x
              
    sigma_mean = sqrt((np.sum(np.power(y[:],2))/samp_size)/(10**(snr/10)))

    sigma = sigma_mean*sigma

    noise = block_diag(sigma)*np.random.normal(y.shape)

    y = y+noise
                   
    return y, x, noise, sigma, outliers]
