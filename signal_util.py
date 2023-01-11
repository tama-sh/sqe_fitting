import numpy as np
import scipy.signal as scisig

def smoothen(data: np.ndarray, dt: float = 1, numtaps: int = 21, smoothing_width: float = 10):
    """Smoothen the data by applying fir filter with zero phase.
    Args:
        data (np.ndarray): data to be smoothened
        dt (float): interval of sampling
        numtaps (int): number of taps used for fir filter
        smoothing_width (float): time width of smoothing, cutoff = 1/smoothing_width
        
    Returns:
        np.ndarray: smoothened data
    """
    fs = 1/dt
    cutoff = 1/smoothing_width
    b = scisig.firwin(numtaps=21, cutoff=cutoff, fs=fs)
    data_lp = scisig.filtfilt(b, 1, data) # apply zero phase filter
    return data_lp

def derivative(y: np.ndarray, x: np.ndarray):
    """Take numerical derivative
    
    Args:
        y (np.ndarray): dependent variable
        x (np.ndarray): independent variable
        
    Returns:
        np.ndarray: array of derivative at (x[1:]+x[:-1])/2 ,the length of array become len(x)-1
    """
    return (y[1:]-y[:-1])/(x[1:]-x[:-1])

def middle_points(x: np.ndarray):
    """Return the middle points of array
    
    Return the middle point of adjacent coordinate (x_i + x_{i+1})/2 as an array
    
    Args:
        x (np.ndarray): array of coordinate
    
    Return:
        np.ndarray: array of middle point
    """
    return 0.5*(x[1:]+x[:-1])

def group_delay(cplx: np.ndarray, omega: np.ndarray):
    """Calculate group delay
    
    Use derivative of cplx instead of unwrapped phase for stability
    
    Args:
        cplx (np.ndarray): complex data with electrical delay
        omega (np.ndarray): angular frequency
        
    Returns:
        np.ndarray: group delay at the middle points of give array of omega ,the length of array become len(cplx)-1
    """
    return -np.imag(derivative(cplx, omega)/middle_points(cplx))
