import numpy as np
import lmfit
import scipy.optimize as sciopt

"""
Ref:
    1. J. Math. Imaging Vision 23, 239 (2005)
    2. Rev. Sci. Instrum. 86, 024706 (2015)
"""

def moment_matrix(x: np.ndarray, y: np.ndarray):
    """Return the matrix of the moments
    
    The matrix of the moments is defined as
    [[Mzz, Mxz, Myz, Mz],
     [Mxz, Mxx, Mxy, Mx],
     [Myz, Mxy, Myy, My],
     [Mz, Mx, My, n]]/n
    where Mxx = \sum_{i=1}^n x_i^2, Mx = \sum_{i=1}^n x_i, and so on.
    
    Args:
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
    Return:
        np.ndarray: matrix of the moments of (x_i, y_i) data
    """
    
    z = x**2 + y**2
    Mxx = np.sum(x**2)
    Myy = np.sum(y**2)
    Mzz = np.sum(z**2)
    Mxy = np.sum(x*y)
    Mxz = np.sum(x*z)
    Myz = np.sum(y*z)
    Mx = np.sum(x)
    My = np.sum(y)
    Mz = np.sum(z)
    n = np.prod(x.shape) # can be used for more than 1d array

    M_mat = np.array([
            [Mzz, Mxz, Myz, Mz],
            [Mxz, Mxx, Mxy, Mx],
            [Myz, Mxy, Myy, My],
            [Mz, Mx, My, n]
            ])/n   # normalized by n, otherwise the fit seems not working well
    
    return M_mat

def Taubin_constraint_matrix(x: np.ndarray, y: np.ndarray):
    """Return Taubin's constraint matrix used in approxymation of objective function
    
    Args:
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
    Return:
        np.ndarray: Tabuin's matrix
    """
    z = x**2 + y**2
    Mx = np.sum(x)
    My = np.sum(y)
    Mz = np.sum(z)
    n = np.prod(x.shape)
    C_mat = np.array([
            [4*Mz, 2*Mx, 2*My, 0],
            [2*Mx, n, 0, 0],
            [2*My, 0, n, 0],
            [0, 0, 0, 0]
            ])/n   # normalized by n
    
    return C_mat

def approx_algebric_circle_fit(x: np.ndarray, y: np.ndarray, method="Pratt"):
    """Gradient-weighted algebraic fit (GRAF) of circle based on approximation of objective function and Newton method
    
    Args:
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
        method (str): approximation method of objective function, either "Pratt" or "Taubin" default is "Pratt"
        
    Return:
        np.ndarray: [A, B, C, D] parameter of circle expressed with A(x^2+y^2)+Bx+Cy+D=0
    """
    norm_factor = np.sqrt(np.mean(x**2+y**2)) # need normalization for convergence of sciopt.newton
    xn, yn = x/norm_factor, y/norm_factor
    M_mat = moment_matrix(xn, yn)
    
    # Define Constraint matrix
    if method == "Pratt":
        C_mat = np.array([
            [0, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2, 0, 0, 0]
        ])
    elif method == "Taubin":
        C_mat = Taubin_constraint_matrix(xn, yn)
    else:
        raise ValueError("method should be 'Pratt' or 'Taubin'")
    
    # Newton method to obtain eta
    def Q(eta):
        return np.linalg.det(M_mat-eta*C_mat)
    eta_opt = sciopt.newton(Q, 0)

    # Get parameters of circle
    eigval, eigvec = np.linalg.eig(M_mat - eta_opt*C_mat)
    idx = np.argmin(abs(eigval)) # retrieve eigen vector corresponding to 0 eigen value
    An, Bn, Cn, Dn = np.ravel(eigvec[:,idx]) # need to retrieve the column vector
    
    # Objective function of Eq.(4.17) in Ref.[2]
    # error = opt_vec @ M_mat @ opt_vec / (B**2+C**2-4*A*D)  # numpy @ operator for matrix multiplication
    
    return np.array([An, norm_factor*Bn, norm_factor*Cn, norm_factor**2*Dn])

def algebric_circle_fit(x: np.ndarray, y: np.ndarray, init_method="Pratt"):
    """"Gradient-weighted algebraic fit (GRAF) of circle

    Args:
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
        init_method (str): approximation method of objective function to estimate initial parameters, either "Pratt" or "Taubin" default is "Pratt"
        
    Return:
        lmfit.MinimizerResult: result of minimiztion
    """
    def residual(pars: lmfit.Parameters, x: np.ndarray, y: np.ndarray):
        parvals = pars.valuesdict()
        A = parvals['A']
        B = parvals['B']
        C = parvals['C']
        D = parvals['D']
        return GRAF_obj_func(A, B, C, D, x, y)
    
    p_init = approx_algebric_circle_fit(x, y, method=init_method)
    
    pars = lmfit.Parameters()
    pars.add_many(('A', 1, False),  # fix with 1
                  ('B', p_init[1]/p_init[0]),
                  ('C', p_init[2]/p_init[0]),
                  ('D', p_init[3]/p_init[0]))
    pars.add('x_c', expr='-B/(2*A)')
    pars.add('y_c', expr='-C/(2*A)'),
    pars.add('r_0', expr='(1/(2*abs(A)))*sqrt(B**2+C**2-4*A*D)')
    
    fitter = lmfit.Minimizer(residual, pars, fcn_args=(x, y))
    rst = fitter.minimize()
    
    # Best fit
    # x_c, y_c, r_0 = rst.params['x_c'], rst.params['y_c'], rst.params['r_0']
    # theta = np.arange(0, 2*np.pi, 2*np.pi/100)
    # rst.best_fit = x_c+r_0*np.cos(theta) + 1j*(y_c+r_0*np.sin(theta))
    
    return rst

def ABCD_to_center_radius(A: float, B: float, C: float, D: float):
    """Convert A, B, C, D parameters of circle to center coodinate and radius
    Args:
        A, B, C, D: Parameters of circle expressed with A(x^2+y^2)+Bx+Cy+D=0
    Returns:
        Tuple[float, float, float]: [x_c, y_c, r_0] where (x_c, y_c) is the coordinate of the center of circle. r_0 is the radius
    """
    x_c = -B/(2*A)
    y_c = -C/(2*A)
    r_0 = (1/(2*abs(A)))*np.sqrt(B**2+C**2-4*A*D)
    return (x_c, y_c, r_0)

def GRAF_obj_func(A: float, B: float, C: float, D: float, x: np.ndarray, y: np.ndarray):
    """GRAF objective function for circle fit
    See. Eq.(4.16) of J. Math. Imaging Vision 23, 239 (2005)
    Note that there is a typo in Eq.(4.16) of the above reference, denominator should not be squred
    
    Args:
        A, B, C, D: Parameters of circle expressed with A(x^2+y^2)+Bx+Cy+D=0
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
    Returns:
        np.ndarray: residuals
    """
    return (A*(x**2+y**2) + B*x + C*y + D)/np.sqrt(4*A*(A*(x**2+y**2) + B*x + C*y + D) + B**2 + C**2 - 4*A*D)

def Pratt_obj_func(A: float, B: float, C: float, D: float, x: np.ndarray, y: np.ndarray):
    """Pratt's approximation of GRAF objective function for circle fit
    See. Eq.(4.17) of J. Math. Imaging Vision 23, 239 (2005)
        
    Args:
        A, B, C, D: Parameters of circle expressed with A(x^2+y^2)+Bx+Cy+D=0
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
        
    Returns:
        np.ndarray: residuals
    """
    return (A*(x**2+y**2) + B*x + C*y + D)/np.sqrt(B**2 + C**2 - 4*A*D)

def Taubin_obj_func(A: float, B: float, C: float, D: float, x: np.ndarray, y: np.ndarray):
    """Taubin's approximation of GRAF objective function for circle fit
    
    Args:
        A, B, C, D: Parameters of circle expressed with A(x^2+y^2)+Bx+Cy+D=0
        x (np.ndarray): array of x coordinates
        y (np.ndarray): array of y coordinates
        
    Returns:
        np.ndarray: residuals
    """
    z = x**2 + y**2
    return (A*z + B*x + C*y + D)/np.sqrt(4*A*(A*z.mean() + B*x.mean() + C*y.mean()) + B**2 + C**2)