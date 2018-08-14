
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

def sisal(Y, p, *args):

# ------------------------------------------------------------------------------
# test for number of required parametres
# ------------------------------------------------------------------------------

    data = ([[1, 2], [3, 4], [5, 6]])    
    L, N = len(data), len(data[0]) 
    p = 2

    # data set size
    if L < p:
        raise ValueError("Insufficient number of columns in y")

# ------------------------------------------------------------------------------
# Set the defaults for the optional parameters
# ------------------------------------------------------------------------------

    # maximum number of quadratic QPs
    MMiters = 80
    spherize = "yes"
    # display only only volume evolution
    verbose = 1
    # soft constraint regularization parameter
    tau = 1
    # Augmented Lagrangian regularization parameter
    mu = p * 1000 / N
    # no initial simplex
    M = 0
    # tolerance for the termination test
    tol_f = 1e-2

# ------------------------------------------------------------------------------
# Local variables
# ------------------------------------------------------------------------------

    # maximum violation of inequalities
    slack = 1e-3
    # flag energy decreasing
    any_energy_decreasing = 0
    # used in the termination test
    f_val_back = inf
    # spherization regularization parameter
    lam_sphe = 1e-8
    # quadractic regularization parameter for the Hesssian
    # Hreg = = mu*I
    lam_quad = 1e-6
    # minimum number of AL iterations per quadratic problem 
    AL_iters = 4
    # flag 
    flaged = 0

# ------------------------------------------------------------------------------
# Read the optional parameters
# ------------------------------------------------------------------------------

    if (length(*args) % 2 == 1):
        raise ValueError("Optional parameters should always go by pairs")
    else:
        for i in range(1, 2, length(*args)-1):
            if (arg[i].upper() == "MM_ITERS"):
                MMiters = arg(i+1)
            elif (arg[i].upper() == "SPHERIZE"):
                spherize = arg(i+1)
            elif (arg[i].upper() == "MU"):
                mu = arg(i+1)
            #check lambda 
            elif (arg[i].upper() == "TAU"):
                tau = arg(i+1)
            elif (arg[i].upper() == "TOLF"):
                tol_f = arg(i+1)
            elif (arg[i].upper() == "MO"):
                M = arg(i+1)
            elif (arg[i].upper() == "VERBOSE"):
                verbose = arg(i+1)
            else:
                raise ValueError("Unrecognized option: "+ arg[i])

# ------------------------------------------------------------------------------
# set display mode
# ------------------------------------------------------------------------------
            
    if(verbose == 3) or (verbose == 4):
        # program in MATLAB turns off all the warnings that could arise
        # this line will do the same in the python program
        warnings.filterwarnings("ignore")
    else:
        # allows all warnings to be on 
        warnings.simplefilter('always')

# ------------------------------------------------------------------------------
# identify the affine space that best represent the data set y
# ------------------------------------------------------------------------------

    my = Y.mean(1)
    Y = Y - np.matlib.repmat(my, 1, N)
    Up, D = svds(Y*Y.getH()/N, p-1)
    # represent y in the subspace R^(p-1)
    Y = Up * Up.getH() * Y
    # lift y
    Y = Y + np.matlib.repmat(my, 1, N)
    # compute the orthogonal component of my
    my_ortho = my - Up * Up.getH() * my
    # define another orthonormal direction
    Up = [Up, my_ortho/sqrt(np.sum(np.power(my_ortho, 2)))]
    sing_values = block_diag(A)

    # get coordinates in R^p
    Y = Up.getH() * Y

# ------------------------------------------------------------------------------
# spherize if requested
# ------------------------------------------------------------------------------

    if spherize.equals('yes'):
        Y = Up * Y
        Y = Y - np.matlib.repmat(my, 1, N)
        C = block_diag(np.divide(1, sqrt(block_diag(D+lam_sphe*np.identity(p-1)))))
        # to find conjugate transpose
        Y = C * Up[:, 0:p-1].conj().T * Y
        # lift
        Y[p, :] = 1
        # normalize to unit norm
        Y = Y/sqrt(p)

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

    if M == 0:
        # Initialize with VCA
        Mvca = VCA(Y,'Endmembers',p)
        M = Mvca
        # expand Q
        Ym = M.mean(1)
        Ym = np.matlib.repmat(Ym, 1, p)
        dQ = M - Ym
        # fraction: multiply by p is to make sure Q0 starts with a feasible
        # initial value.
        M = M + p*dQ;
    else:
        # Ensure that M is in the affine set defined by the data
        M = M - np.matlib.repmat(my, 1, p)
        M = Up[:, 0:p-1] * Up[:, 0:p-1].conj().T * M
        M = M + np.matlib.repmat(my, 1, p)
        M = Up.getH() * M    # represent in the data subspace
        # is sherization is set
        if spherize.equals('yes'):
            M = Up * M - np.matlib.repmat(my, 1, p)
            M = C * Up[:, 0:p-1].conj().T * M
            #  lift
            M[p, :] = 1
            # normalize to unit norm
            M = M/sqrt(p)

    Q0 = inv(M)
    Q = Q0

    # plot  initial matrix M
    if(verbose == 2) or (verbose == 4):
        plt.setp(0, 'Unit', 'Pixels')

        # get figure 1 handler
        H_1 = plt.figure();
        pos1 = plt.getp(H_1, 'Position')
        pos1[1]=50;
        pos1[2]=100+400;
        plt.setp(H_1,'Position', pos1)

        hold()
        M = inv(Q);
        p_H[1] = plot(Y[1,:],Y[2,:], '.');
        p_H[2] = plot(M[1,:], M[2,:],'ok');

        # leg_cell = cell(1);
        # leg_cell{1} = 'data points';
        # leg_cell{end+1} = 'M(0)';
        
        plt.title('SISAL: Endmember Evolution')

    end

# ------------------------------------------------------------------------------
# build constraint matrices
# ------------------------------------------------------------------------------

    AAT = np.kron(Y.getH() * Y, np.identity(p))
    B = np.kron(np.identity(p), np.ones(1, p))  # size p^2xp^2
    qm = np.sum(inv(Y * Y.getH()) * Y, 2)       # size pxp^2

    H = lam_quad*np.identity(p**2)
    F = H + mu * AAT    # equation (11) of [1]
    IF = inv(f)

    # auxiliar constant matrices
    G = IF * B.getH() * inv(B * IF * B.getH())
    qm_aux = G * qm
    G = IF - G * B * IF

# ------------------------------------------------------------------------------
#           Main body- sequence of quadratic-hinge subproblems
# ------------------------------------------------------------------------------

    # initializations
    Z = Q * Y
    Bk = 0 * Z

    for k in range(MMiters + 1):
        
        IQ = inv(Q)
        g = -IQ.getH()
        g = np.reshape(g, (np.product(g.shape)))

        baux = H * np.reshape(Q, (np.product(Q.shape),)) - g

        q0 = np.reshape(Q, (np.product(Q.shape),))
        Q0 = Q

        # display the simplex volume
        if verbose == 1:
            if spherize.equals('yes'):
                # unscale
                M = IQ * sqrt(p)
                # remove offset
                M = M[1:p-1, :]
                # unspherize
                M = Up[:, 1:p-1] * IC * M
                # sum ym
                M = M + np.matlib.repmat(my, 1, p)
                M = Up.getH() * M
            else:
                M = IQ
        print("\n iter = %d, simplex volume = %4f  \n", k,
              1/np.absolute(np.linalg.det(M)))


        # Bk = 0*Z
        if k == MMiters:
            AL_iters = 100;
            # Z=Q*Y;
            # Bk = 0*Z;

        # initial function values (true and quadratic)
        # f0_val = -log(abs(det(Q0)))+ tau*sum(sum(hinge(Q0*Y)));
        # f0_quad = f0_val; % (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);

        while 1 > 0:
            q = np.reshape(Q, (np.product(Q.shape)))
            # initial function values (true and quadratic)
            f0_val = -np.log(np.absolute(np.linalg.det(Q)))+ tau*np.sum(np.sum(hinge(Q*Y)))
            f0_quad = (q-q0).getH()*g+1/2*(q-q0).getH()*H*(q-q0) + tau*np.sum(np.sum(hinge(Q*Y)))
            for i in range (1, AL_iters):
                #-------------------------------------------
                # solve quadratic problem with constraints
                #-------------------------------------------
                dq_aux= Z+Bk              # matrix form
                dtz_b = dq_aux*Y.getH()
                dtz_b = dtz_b[:]
                b = baux+mu*dtz_b         # (11) of [1]
                q = G*b+qm_aux            # (10) of [1]
                # might need different syntax 
                Q = np.reshape(q,p,p)

                #-------------------------------------------
                # solve hinge
                #-------------------------------------------
                Z = soft_neg(Q*Y -Bk,tau/mu);
            
                 # norm(B*q-qm)

                #-------------------------------------------
                # update Bk
                #-------------------------------------------
                Bk = Bk - (Q*Y-Z);
                if verbose == 3 ||  verbose == 4:
                    print("\n ||Q*Y-Z|| = %4f \n",LA.norm(Q*Y-Z,'fro'))
                if verbose == 2 || verbose == 4:
                    M = inv(Q)
                    plt.plot(M[1,:], M[2,:],'.r')
                    if is not flaged
                        p_H[3] = plt.plot(M[1,:], M[2,:],'.r')
                        # leg_cell{end+1} = 'M(k)'
                        flaged = 1
            f_quad = (q-q0).getH()*g+1/2*(q-q0).getH()*H*(q-q0)+tau*np.sum(np.sum(hinge(Q*Y)))
            if verbose == 3 ||  verbose == 4:
                print("\n MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_quad = %2.4f,  \n",...
                k,i, f0_quad,f_quad)
                
            f_val = -nplog(np.absolute(np.linalg.det(Q)))+tau*np.sum(np.sum(hinge(Q*Y)))
            if f0_quad >= f_quad    # quadratic energy decreased:
                while  f0_val < f_val:
                    if verbose == 3 ||  verbose == 4:
                        print("\n line search, MMiter = %d, AL_iter, = % d,  f0 = %2.4f, f_val = %2.4f,  \n",...
                        k,i, f0_val,f_val)
                    # do line search
                    Q = (Q+Q0)/2
                    f_val = -np.log(np.absolute(np.linalg.det(Q)))+tau*np.sum(np.sum(hinge(Q*Y)))
                break

        if verbose == 2 || verbose == 4:
            p_H[4] = plt.plot(M[1,:], M[2,:],'*g');
            # leg_cell{end+1} = 'M(final)';
            # legend(p_H.getH(), leg_cell);

        return M, Up, my, sing_values 

    


                
                
                
                
            
            
        
        
    
    














