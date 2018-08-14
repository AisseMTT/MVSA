#
#
# Minimum Volume Simplex Analysis: A fast Algorithm to Unmix Hyperspectral Data
# This code has been translated from MATLAB into Python 
# This file contains only the MVSA Algorithm
#
# MVSA Estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional
# simplex of minimum volume containing the vectors [y_1,...y_N], under the
# assumption that y_i belongs to a (p-1)  dimensional affine set. Thus,
# any vector y_i   belongs  to the convex hull of  the columns of M; i.e.
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


# ------------------------------------------------------------------------------
# test for number of required parametres
# ------------------------------------------------------------------------------
def mvsa(Y, p, *args):
    # This part of the MATLAB code is already done by Python automatically
    
    # if(nargin - length(*args)) != 2:
        # raise ValueError("Wrong number of required parameters")

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
    MMiters = 4
    spherize = "yes"
    # display only MVSA warnings
    verbose = 1
    # spherization regularization parameter
    _lambda = 1e-10
    # quadractic regularization parameter for the Hesssian
    # Hreg = = mu*I+H
    mu = 1e-6
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
            elif (arg[i].upper() == "LAMBDA"):
                lmda = arg(i+1)
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
                
    if(verbose == 0) or (verbose == 1):
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
        C = block_diag(np.divide(1, sqrt(block_diag(D+lmda*np.identity(p-1)))))
        # to find conjugate transpose
        Y = C * Up[:, 0:p-1].conj().T * Y
        # lift
        Y[p, :] = 1
        # normalize to unit norm
        Y = Y/sqrt(p)
        
# ------------------------------------------------------------------------------
#  Initialization
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

# ------------------------------------------------------------------------------
# build constraint matrices
# ------------------------------------------------------------------------------

    # inequality matrix
    A = np.kron(Y.getH(), np.identity(p))
    # equality matrx
    E = np.kron(identity(p), np.ones(1, p))
    # equality independent vector
    qm = np.sum(inv(Y*Y.getH())*Y, 2) 

# ------------------------------------------------------------------------------
# sequence of QPs - main body
# ------------------------------------------------------------------------------

    for k in range(MMiters + 1):
        # make initial point feasible
        M = inv(Q)
        Ym = M.mean(1)
        Ym = np.matlib.repmat(Ym, 1, p)
        dW = M - Ym
        count = 0
        while np.sum(np.sum(inv(M) * Y < 0)) > 0:
            M = M + 0.01*dW
            count = count + 1
            if count > 100:
                if verbose:
                    print("Could not make M feasible after 100 expansions")
                break
        Q = inv(M)
        # gradient of -log(abs(det(Q)))
        g = -(M.getH())
        g = np.reshape(g, (np.product(g.shape)))

        # quadractic term (mu*I+diag(H))
        H = mu * np.identity(p**2) + block_diag(np.power(g, 2))
        q0 = np.reshape(Q, (np.product(Qd.shape),))
        Q0 = Q
        f = g - H * q0

        # initial function values (true and quadratic)
        f0_val = -np.log(np.absolute(np.linalg.det(Q0)));
        f0_quad = f0_val;     # (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);

        # anergy decreasing in this quadratic problem
        energy_decreasing = 0

        #%%%%%%%%%%%%%% QP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

        # Here we make the function that completes the purpose of quadprog
        def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
            qp_G = .5 * (P + P.T)   # make sure P is symmetric
            qp_a = -q
        if A is not None:
            qp_C = -numpy.vstack([A, G]).T
            qp_b = -numpy.hstack([b, h])
            meq = A.shape[0]
        else:       # no equality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0
        return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
    
        P = q
        q = f_dummy
        G = exitflag
        h = output 
        
        quadprog_solve_qp(P, q, G, h)

        if  exitflag < 0:    # quadprog did not converge
            if verbose:
                print("\niter = %d, quadprog did not converge: exitflag = %d \n"
                      ,+ K, + exitflag)
            if k == MMiters:
                if not any_energy_decreasing:
                    if verbose:
                        print("\n outputing the VCA solution\n")
                    Q = inv(Mvca)
                    q = np.reshape(Q, (np.product(Q.shape)))
                else:
                    if verbose:
                        print("\n outputing the solution" +
                              "of the previous iteration\n")
                    Q = Q0
                    q = np.reshape(Q, (np.product(Q.shape)))
            else:
                # run again with a larger mu
                Q = Q0;
                q = np.reshape(Q, (np.product(Q.shape)))
                mu = 1e-2
        elif exitflag == 0:      # Number of iterations exceeded options.MaxIter.
            # compute  energy of the quadratic approximation
            f_quad = f0_val + (q-q0).conj().T * g+1/2 * (q-q0).conj().T * H * (q-q0)
            if verbose:
                print("\n iterations exceeded:" +"iter = %d, f0_quad = %2.4f,"+ 
                      "f_quad = %2.4f, iter(QP) = %d \n'" + k,+ f0_quad,+ f_quad
                      , +output.iterations)
            # test for energy decreasing and feasibility
            if (f0_quad > f_quad) and (np.sum(np.sum(Q * Y < -slack)) == 0):
                if verbose:
                    print("\n test for quadratic energy decreasing" +
                          "and feasibility PASSED\n")
                    # there will be surely an energy decreasing between for Q
                    # between the current Q and Q0
                    energy_decreasing = 1
                    any_energy_decreasing = 1
                else:
                    if verbose:
                        print("\n test for quadratic energy decreasing FAILED\n")
                        print("\n Incremente H\n")
                    # increment H
                    Q = Q0
                    q = np.reshape(Q, (np.product(Q.shape)))
                    mu = 1e-2
                    
        # energy_decreasing == 1 means that  although exitflag ~= 1, the
        # energy of the quadratic approximation decreased.
        # exiflaf == 1 implies that the energy of the quadratic approximation
        # decreased.
        if (energy_decreasing == 1) or (exitflag == 1):
            Q = np.reshape(q,p,p)
            # f_bound
            f_val = -np.log(np.absolute(np.linalg.det(Q)));
            if verbose:
                print("\n iter = %d, f0 = %2.4f, f = %2.4f, exitflag = %d", +
                      "iter(QP) = %d \n',...", + k, f0_val,f_val, +
                      exitflag,output.iterations)
            # line search
            counts = 1
            while (f0_val < f_val):
                # Q and Q0 are in a convex set and f(alpha*Q0+(1-alpha)Q) <
                # f(Q0)  for some alpha close to zero
                Q = (Q+Q0)/2
                f_val = -np.log(np.absolute(np.linalg.det(Q)));
                if verbose:
                    print("\n doing line search: counts = %d, f0 = %2.4f, ", +
                          "f = %2.4f\n', ...", + counts, f0_val, f_val)
                counts = counts + 1
                if counts > 20:
                    print("\n something wrong with the line search\n")
                    if k == MMiters:
                        if not energy_decreasing:
                            print("\n outputing the VCA solution\n")
                            Q = inv(Mvca)
                        else:
                            print("\n outputing the solution of the previous iteration\n")
                            Q = Q0
                    else:
                        # run again with a larger mu
                        Q = Q0
                        mu = 1e-2
            energy_decreasing = 1
        # termination test
        if energy_decreasing:
            if np.absolute((f_val_back-f_val)/f_val) < tol_f:
                if verbose:
                    print("\n iter: = %d termination test PASSED \n ",+ k)
                break
            f_val_back = f_val
            break
        break
    
        return M, Up, my, sing_values
            
    if spherize.equals('yes'):
        M = inv(Q)
        # refer to the initial affine set
        # unscale
        M = M*sqrt(p)
        # remove offset
        M = M[0:p-1, :]
        # unspherize
        M = Up[:, 0:p-1] * block_diag(sqrt(block_diag(D+lmda*np.identity(p-1)))) * M
        # sum ym
        M = M + np.matlib.repmat(my, 1, p)
    else:
        M = Up*inv(Q)

        
        


    
            
                            
                        

                    
            

        



        

    
