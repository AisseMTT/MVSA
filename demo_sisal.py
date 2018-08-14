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

# This demo illustrates how to run  SISAL algorithm [1]

## beginning of the demo
verbose = 1

#--------------------------------------------------------------------------
#        Simulation parameters
#-------------------------------------------------------------------------
# p                         -> number of endmembers
# N                         -> number of pixels
# SNR                       -> signal-to-noise ratio (E ||y||^2/E ||n||^2) in dBs
# SIGNATURES_TYPE           -> see below
# L                         -> number of bands (only valid for SIGNATURES_TYPE = 5,6)
# COND_NUMBER               -> conditioning number of the mixing matrix (only for SIGNATURES_TYPE = 5,6)
# DECAY                     -> singular value decay rate
# SHAPE_PARAMETER           -> determines the distribution of spectral
#                              over the simplex (see below)
# MAX_PURIRY                -> determines the maximum purity of  the
#                              mixtures.  If  MAX_PURIRY < 1, ther will be
#                              no pure pixels is the data

# SIGNATURES_TYPE;
#  1 - sampled from the USGS Library
#  2 - not available 
#  3 - random (uniform)
#  4 - random (Gaussian)
#  5 - diagonal with conditioning number COND_NUMBER and DECAY exponent
#  6 - fully populated with conditioning number COND_NUMBER and DECAY exponent

# NOTE: For SIGNATURES_TYPE 5 or 6, the difficulty of the problem is
# determined by the parameters
# COND_NUMBER
# DECAY


# Souces are Dirichlet distributed
# SHAPE_PARAMETER ;
#   = 1   -  uniform over the simplex
#   > 1   -  samples moves towards the center
#            of the simplex, corresponding to highly mixed materials and thus
#            the unmixing is  more difficult
#   ]0,1[ -  samples moves towards the facets
#            of the simplex. Thus the unmixing is easier.


#
#--------------------------------------------------------------------------
#        SELECTED PARAMETERS FOR AN EASY PROBLEM
#-------------------------------------------------------------------------

SIGNATURES_TYPE = 1        # Uniform in [0,1]
p = 3                      # number of endmembers
N = 5000                   # number of pixels
SNR = 50                   # signal-to-noise ratio (E ||y||^2/E ||n||^2) in dBs
L = 200                    # number of bands (only valid for SIGNATURES_TYPE = 2,3)
# COND_NUMBER  = 1           # conditioning number (only for SIGNATURES_TYPE = 5,6)
# DECAY = 1                  # singular values decay rate  (only for SIGNATURES_TYPE = 5,6)
SHAPE_PARAMETER = 1        # uniform over the simplex
MAX_PURIRY = 0.8           # there are pure pixels in the data set
OUTLIERS = 0               # Number of outliers in the data set

#--------------------------------------------------------------------------
#        Begin the simulation
#-------------------------------------------------------------------------
if SIGNATURES_TYPE == 1:
    random.seed(5)

    # open and store the workspace variables in Python 
    mat_file = sio.loadmat('USGS_1995_Library.mat')
    datalib = mat_file['datalib']
    names = mat_file['names']

    wavlen = datalin[:, 1]  # Wavelengths in Microns
    L, n_materials = datalib.shape
    # select randomly
    sel_mat = 4 + np.random.permutation(n_materials-4)
    sel_mat = sel_mat[1:p]
    M = datalib[:, sel_mat]
    # print selected endmembers
#       fprintf('endmembers:\n')
#       for i=1:p
#           aux = names(sel_mat(i),:);
#           fprintf('%c',aux);
#           st(i,:) = aux;
#       end
elif SIGNATURES_TYPE == 2:
    raise ValueError("type not available")
elif SIGNATURES_TYPE == 3:
    M = np.random.uniform(L,p)
elif SIGNATURES_TYPE == 4:
    M = np.random.normal(L, p)
elif SIGNATURES_TYPE == 5:
    L = p
elif SIGNATURES_TYPE == 6:
    L = p
    M = block_diag(np.power(np.linespace(1, (1/(COND_NUMBER)**(1/DECAY)),p)))
    A = np.random.normal(p)
    U, D, V = np.linalg.svd(A)
    M = U * M * V.getH()
else:
    raise ValueError("wrong signatute type")

#--------------------------------------------------------------------------
#        Set noise parameters (to be used in spectMixGen function)
#-------------------------------------------------------------------------

# white_noise = [0 1 1];     % white noise
# % non-white noise parameters
# eta   = 10;                % spread of the noise shape
# level = 10;                % floor lever
# gauss_noise = [level L/2 eta]; % Gaussian shaped noise centered at L/2 with spread eta
# % and floor given by level
# rect_noise  = [level L/2 eta]; % Rectangular shaped noise centered at L/2 with spread eta
# % and floor given by level

#
#--------------------------------------------------------------------------
#        Generate the data set
#-------------------------------------------------------------------------
#
#   Sources are Diriclet distributed (shape is controled by 'Source_pdf' and
#   'pdf_pars': 'Source_pdf' = 'Diri_id' and 'pdf_pars' = 1 means a uniform
#   density over the simplex).  The user may change the parameter to
#   generate other shapes. Mixtures are aldo possible.
#
#   'max_purity' < 1 means that there are no pure pixels
#

[Y,x,noise] = spectMixGen(M,N,'Source_pdf', 'Diri_id','pdf_pars',SHAPE_PARAMETER,...
    'max_purity',MAX_PURIRY*ones(1,p),'no_outliers',OUTLIERS, ...
    'violation_extremes',[1,1.2],'snr', SNR, ...
    'noise_shape','uniform')

#--------------------------------------------------------------------------
#        Remove noise  (optional)
#-------------------------------------------------------------------------
#   noise_hat = estNoise(Y);
#   Y = Y-noise_hat;
#   clear noise_hat

#--------------------------------------------------------------------------
#       Project  on the  affine set defined by the data in the sense L2
#-------------------------------------------------------------------------
#
#   The application of this projection ensures that the data is in
#   an affine set.
#
#   Up is an isometric matrix that spans the subspace where Y lives
Y, Up, my, sing_val = dataProj(Y, p, 'proj_type', 'affine')

#--------------------------------------------------------------------------
#        Degree of Difficulty of the problem
#-------------------------------------------------------------------------
# compute original subspace
sing_vects = svds(M, p)

# Condition number gives an idea of the difficulty in inferring
# the subspace
#printf('Conditioning number of M = %2f \n', sing_vects(1)/sing_vects(end))
# fprintf('\n Hit any key: \n ');
# pause;

Cx = Up.getH()*(M*x)*(M*x).getH()*Up/N
Cn = Up.getH()*noise*noise.getH()*Up/N
U, D = np.linalg.svd(Cx)

# compute the SNR along the direction corresponding the smaller eigenvalue

LOWER_SNR = D[p, p]/(Up[:,0:p].conj()T*Cn*U[:, p])
print('\nSNR along the signal smaller eigenvalue = %f \n', LOWER_SNR)
if LOWER_SNR < 20
   print('\nWARNING: This problem is too hard and the results may be inaccurate \n')

#--------------------------------------------------------------------------
#         ALGORITHMS
#-------------------------------------------------------------------------

# Foe each algorithm, define the maximum  number of endmembers for each
# algorithm. Beyond  this number, the algorithm take "too much time"

# max_sisal_p = inf;   work in any case
max_mvsa_p = 0    #   do nor run mvsa
max_mves_p = 0    #   do not run mves
max_vca_p = 100   #   work in any case

#--------------------------------------------------------------------------
#         SISAL[1] -  Simplex identification via split augmented Lagrangian
#-------------------------------------------------------------------------

# set the hinge regularization parameter
tau = 10
A_est = sisal(Y, p, 'spherize', 'yes','MM_ITERS',40, 'TAU',tau, 'verbose',2)
# drawnow;
# t[1] = toc
Msisal =  Up.getH() * A_est

#--------------------------------------------------------------------------
#         VCA [5] - Vertex component analysis
#-------------------------------------------------------------------------

Ae, indice, ys = VCA(Y,'Endmembers',p)
Mvca = Up.getH()*Ae
# stop timer
# t[2] = toc;

#--------------------------------------------------------------------------
#         Project the original mixing matxix and the data set the
#         identified affine set.
#-------------------------------------------------------------------------

Mtrue = Up.getH()*M
Y=Up.getH()*Y

#--------------------------------------------------------------------------
#        Display the results
#-------------------------------------------------------------------------

# selects axes  to display

I = 1
J = 2
K = 3

# canonical orthogonal directions
E_I = np.identity(p)

v1 = E_I[:,I]
v2 = E_I[:,J]
v3 = E_I[:,K]

# original axes

Q = inv(Mtrue)
# v1 = Q(I,:)';
# v2 = Q(J,:)';
# v3 = Q(K,:)';

Y = [v1 v2 v3].conj().T*Y
m_true = [v1 v2 v3].conj().T*Mtrue
m_sisal = [v1 v2 v3].conj().T*Msisal

# legend
# leg_cell = cell[1]
# leg_cell{end} = 'data points'
H_2=plt.figure();
plot(Y[1,:],Y[2,:],'k.','Color',[ 0 0 1])

# hold on;
plot(m_true[1,[1:p 1]], m_true[2,[1:p 1]],'*', 'Color',[0 0 0])
# leg_cell{end +1} = 'true';


plot(m_sisal[1,1:p], m_sisal[2,1:p],'S', 'Color',[1 0 0])
# leg_cell{end +1} = 'SISAL';

if p <= max_vca_p:
    m_vca  = [v1 v2 v3].conj().T*Mvca
    plot(m_vca[1,[1:p 1]], m_vca[2,[1:p 1]],'p', 'Color',[0 0 0])
    # leg_cell{end +1} = 'VCA';


# xlabel('v1''*Y'),ylabel('v2''*Y');
# legend(leg_cell)
plt.title('Endmembers and data points (2D projection)')



print('\nTIMES (sec):\n SISAL = %3.2f, \n VCA = %3.2f\n', t(1), t(2))

#--------------------------------------------------------------------------
#        Display errors
#-------------------------------------------------------------------------

# alignament
# soft

angles = np.divide(Mtrue.getH()*Msisal,(np.matlib.repmat(sqrt(np.sum(np.multiply(np.power(Mtrue,2)),p,1).conj().T,(np.matlib.repmat(sqrt(np.sum(np.power(Msisal,2)),p,1)))))))
P = np.zeros(p)
for i in range(0, p + 1):
    dummy, j = np.amax(angles[i,:])
    P[j,i] = 1
    angles[:,j] = -inf

# permute colums
Msisal = Msisal*P;

SISAL_ERR = LA.norm(Mtrue-Msisal, 'fro')/LA.norm(Mtrue, 'fro')

angles = Mtrue.getH()*np.divide(Mvca, np.multiply(np.matlib.repmat(sqrt(np.sum(np.power(Mtrue,2)),p,1).conj().T), (np.matlib.repmat(sqrt(np.sum(np.power(Mvca,2)))),p,1)));
P = np.zeros(p);
for i in range (0, p+1):
    dummy, j = np.amax(angles[i,:])
    P[j,i] = 1;
    angles[:,j] = -inf
# permute colums
Mvca = Mvca*P

VCA_ERR = LA.norm(Mtrue-Mvca, 'fro')/LA.norm(Mtrue, 'fro')

st = print("ERROR(mse): \n SISAL = %f\n VCA = %f\n", SISAL_ERR, VCA_ERR)
print("\n " + str(st) + "\n")


#--------------------------------------------------------------------------
#        Plot signatures
#-------------------------------------------------------------------------
# Choose signatures

# leg_cell = cell(1);
H_3 = plt.figure()
# hold on
# clear p_H;

# plot signatures
p_H[1] = plot(1:L, (Up*Mtrue[:,1]).conj().T,'k')
# leg_cell{end} = 'Mtrue';
p_H[2] = plot(1:L,Up*Msisal[:,1]).conj().T,'r')
# sleg_cell{end+1} = 'Msisal';

for i in range(1, p+1):
    plot(1:L,(Up*Mtrue[:,i]).conj().T,'k')
    plot(1:L,(Up*Msisal[:,i]).conj().T,'r')
    if p<= max_mves_p:
        plot(1:L,(Up*Mves[:,i]),'g')
    if p<= max_mvsa_p:
        plot(1:L,(Up*Mvsa[:,i]).conj().T,'b')

# legend(leg_cell)
plt.title("First endmember")
# xlabel('spectral band')

pos2 = get(H_2,'Position');
pos2(1)=50;
pos2(2)=1;
set(H_2,'Position', pos2)

pos3 = get(H_3,'Position');
pos3(1)=600;
pos3(2)=100+400;
set(H_3,'Position', pos3)



