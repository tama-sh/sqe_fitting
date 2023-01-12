import numpy as np
import scipy.signal as scisig
from .util import percentile_range_data

def smoothen(data: np.ndarray, t = 1, numtaps: int = 21, smoothing_width: float = 10):
    """Smoothen the data by applying fir filter with zero phase.
    Args:
        data (np.ndarray): data to be smoothened
        t (float): sampling time or interval of sampling
        numtaps (int): number of taps used for fir filter
        smoothing_width (float): time width of smoothing, cutoff = 1/smoothing_width
        
    Returns:
        np.ndarray: smoothened data
    """
    if isinstance(t, np.ndarray):
        dt = t[1] - t[0]
    elif isinstance(t, (float, int)):
        dt = t
    else:
        raise ValueError('t should be either np.ndarray or float')
    
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

def find_peaks(data, x=1., height=None, distance=None, prominence=None, width=None, **kwargs):
    """Find peaks from data, wrapper function of scipy.signal.find_peaks
    Return values are same as scipy.signal.find_peaks
    
    Args:
        data (np.ndarray): Data with peaks
        x (np.ndarray): x-axis of data
        height (float): Threshold of height in the unit of standard deviation of baseline noise (e.g. height=10 means that peak larger than 10\sigma from baseline would be detected)
        distance (float): Minimum distance between the peaks, distance in the same unit of frequency
        kwargs: other key-word arguments to be passed to scipy.signal.find_peaks
    Returns:
        peaks (np.ndarray): peak indices
        properties (dict): properties of peaks
    """
    baseline_data = percentile_range_data(data, (0, 0.5)) # baseline estimation by using 0-0.5 percentile of data
    mu = np.mean(baseline_data)
    sigma = np.sqrt(np.mean(baseline_data**2)-mu**2)
    
    if height is None:
        height_scipy = None
    elif isinstance(height, (float, int)) or isinstance(height, np.ndarray):
        height_scipy = mu + height*sigma
    else:
        height_scipy = [mu + h*sigma for h in height]
        
    if prominence is None:
        prominence_scipy = None
    elif isinstance(height, (float, int)) or isinstance(prominence, np.ndarray):
        prominence_scipy = prominence*sigma
    else:
        prominence_scipy = [p*sigma for p in prominence]
        
    if isinstance(x, np.ndarray):
        dx = x[1] - x[0]
    elif isinstance(x, (float, int)):
        dx = x
    else:
        raise ValueError('x should be either np.ndarray or float')
    if distance is None:
        distance_scipy = None
    else:
        distance_scipy = max(1, int(distance/dx))
    if width is None:
        width_scipy = None
    else:
        width_scipy = max(1, int(width/dx))
    
    kwargs.update({'height': height_scipy, 'distance': distance_scipy, 'prominence': prominence_scipy, 'width': width_scipy})
    peaks, properties = scisig.find_peaks(data, **kwargs)
    return peaks, properties

def find_major_axis(cplx: np.ndarray):
    """Find major axis from complex data
    
    Find major axis from complex data and return the angle indicating the direction of major axis
    
    Args:
        cplx (np.ndarray(complex)): IQ data as complex number
        
    Return:
        theta: angle of the major axis
        
    """
    X = np.stack((cplx.real, cplx.imag), axis=0)
    cov = np.cov(X, bias=0)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argmin(eigval)
    theta = np.arctan2(eigvec[idx, 0], eigvec[idx, 1])
    return theta
