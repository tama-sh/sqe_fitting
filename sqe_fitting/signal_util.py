import numpy as np
import scipy.signal as scisig
from copy import deepcopy
from .util import intersect_indices
from scipy.interpolate import UnivariateSpline

def smoothen(data: np.ndarray, t = 1, numtaps: int = 11, smoothing_width: float = 10):
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
    b = scisig.firwin(numtaps=numtaps, cutoff=cutoff, fs=fs)
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

def extract_edge_data(data, x=None, edge_width=None, edge_side='both'):
    if x is None:
        x = np.arange(0, len(data))
    if edge_width is None:
        edge_width = 0.1*(x[-1]-x[0]) # 10% of data are userd as edge
    
    if edge_side == 'both':
        if np.isscalar(edge_width):
            edge_width = np.array([edge_width]*2)
        else:
            edge_width = np.array(edge_width)
    
        left_data_indices = np.where(x <= x[0]+edge_width[0])[0]
        right_data_indices = np.where(x >= x[-1]-edge_width[1])[0]
        edge_x = np.concatenate([x[left_data_indices], x[right_data_indices]])
        edge_data = np.concatenate([data[left_data_indices], data[right_data_indices]])
    elif edge_side == 'left':
        left_data_indices = np.where(x <= x[0]+edge_width)[0]
        edge_x = x[left_data_indices]
        edge_data = data[left_data_indices]
    elif edge_side == 'right':
        right_data_indices = np.where(x >= x[-1]-edge_width)[0]
        edge_x = x[right_data_indices]
        edge_data = data[right_data_indices]
    else:
        raise ValueError("edge_side must be 'left', 'right' or 'both.")
    return edge_data, edge_x

def fit_background_at_edge(data, x=None, edge_width=None, edge_side='both'):
    edge_data, edge_x = extract_edge_data(data, x=x, edge_width=edge_width, edge_side=edge_side)
    spline = UnivariateSpline(edge_x, edge_data)
    return spline

def estimate_background_noise_from_edge(data, x=None, edge_width=None, edge_side='both'):
    edge_data, edge_x = extract_edge_data(data, x=x, edge_width=edge_width, edge_side=edge_side)
    spline = UnivariateSpline(edge_x, edge_data)
    sigma = np.sqrt(np.mean((edge_data - spline(edge_x))**2))
    mu = np.mean(edge_data)
    return mu, sigma

def find_peaks(data, x=1., height=None, distance=None, prominence=None, width=None, background_edge_width=None, **kwargs):
    """Find peaks from data, wrapper function of scipy.signal.find_peaks
    Return values are same as scipy.signal.find_peaks
    
    Args:
        data (np.ndarray): Data with peaks
        x (np.ndarray): x-axis of data
        height (float): Threshold of height normalized baseline noise (e.g. height=10 means that peak larger than 10\sigma from baseline would be detected)
        distance (float): Minimum distance between the peaks, distance in the same unit of frequency
        prominence (float): Prominence normalized by normalized by baseline_noise
        width: Peak width
        background_edge_width: The width of edge used for background noise estimation
        kwargs: other key-word arguments to be passed to scipy.signal.find_peaks
    Returns:
        peaks (np.ndarray): peak indices
        properties (dict): properties of peaks
    """
    if isinstance(x, np.ndarray):
        dx = x[1] - x[0]
    elif isinstance(x, (float, int)):
        dx = x
    else:
        raise ValueError('x should be either np.ndarray or float')
    
    if background_edge_width is None:
        default_edge_data_ratio = 10
        background_edge_width = len(data)//default_edge_data_ratio
    else:
        background_edge_width = background_edge_width//dx
    mu, sigma = estimate_background_noise_from_edge(data, edge_width=background_edge_width)
    
    if height is None:
        height_scipy = None
    elif np.isscalar(height):
        height_scipy = mu + height*sigma
    else:
        height_scipy = [mu + h*sigma for h in height]
        
    if prominence is None:
        prominence_scipy = None
    elif np.isscalar(prominence):
        prominence_scipy = prominence*sigma
    else:
        prominence_scipy = [p*sigma for p in prominence]

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

def pelt_linear(y, penalty, x=None):
    """Pruned Exact Linear Time (PELT) changepoints detection for linear regression
    Ref: https://arxiv.org/abs/1101.1438
    - linear regression: maximum log-likelihood loss function -> least square
    - For maximum likelihood loss function: K = 0

    Args:
        y (np.ndarray): 1-dimensional array of data
        penalty (float): penalty of broken line
        x (np.ndarray): 1-dimensional array of data
    return:
        list: change points
    """
    N = len(y)
    if x is None:
        x = np.arange(N)

    change_points = {}
    change_points[-1] = []
    cost = np.zeros(N+1)
    cost[-1] = -penalty
    bending_candidates = [-1]  # collection of bending points

    for n in np.arange(N):
        cfit = []
        f = []
        for k in bending_candidates:
            if len(y[k+1:n+1]) <= 2:
                cfit.append(0)  # residuals = 0 for less than or equal to 2 samples
            else:
                _, res, *_ = np.polyfit(x[k+1:n+1], y[k+1:n+1], 1, full=True)
                cfit.append(res[0]) # residuals as a measure of fit
            f.append(cost[k])
        total_cost_list = np.array(f) + np.array(cfit) + penalty  # Add penalty of bending
        
        min_index = np.argmin(total_cost_list)
        cost[n] = total_cost_list[min_index]

        best_bending_point = bending_candidates[min_index]
        change_points[n] = deepcopy(change_points[best_bending_point]) + [best_bending_point]
        bending_candidates.append(n)
 
    rst = change_points[N-1]
    return rst[1:]

# Guess utlility
def guess_peak_or_dip(data):
    """ Guess data has a peak or a dip
        Args:
            data (np.ndarray): data
        Return:
            bool: True if data looks like having peaks
    """
    return np.median(data) < (np.min(data) + np.ptp(data)/2)

def guess_linewidth_from_peak(freq, data, r=2):
    """ Estimate line width sigma (half of FWHM) from peak, for LorentzianModel
    
        Args:
            freq (np.ndarray): frequency
            data (np.ndarray): data with peak of dip (peak, dip is automatically estimated)
            r (float): Theshold value to estimate the peak. The peak width is estimated from the range of data where data > (1-1/r)*max(data)
            
        Return:
            float: sigma (half of FWHM)
    """
    length = len(data)
    idx_c = np.argmax(data[1:-1]) + 1 # avoid the peak placing at borders
    
    ptp = np.ptp(data)
    max_val = data[idx_c]
    cond = data > (max_val - ptp/r)

    i = 0
    while (idx_c+i+1 < (length-1) and cond[idx_c+i+1]):
        i += 1
    j = 0
    while (idx_c-(j+1) > 0 and cond[idx_c-(j+1)]):
        j += 1
    idx_l = idx_c - j
    idx_r = idx_c + i
    
    # redefine r parameter
    r = ptp/(data[idx_c]-0.5*(data[idx_l-1]+data[idx_r+1]))
    return np.sqrt(r-1)*(freq[idx_r+1] - freq[idx_l-1])/2

def estimate_phase_offset(freq1, freq2, cplx1, cplx2):
    idx1, idx2 = intersect_indices(freq1, freq2)
    theta21 = np.angle(np.mean(cplx2[idx2]*np.conj(cplx1[idx1])))
    return theta21

def combine_sparameter_with_phase_offset(freq_list, cplx_list, is_offset_equal=False):
    phase_diff_list = []
    for i in range(len(freq_list)-1):
        freq1 = freq_list[i]
        freq2 = freq_list[i+1]
        cplx1 = cplx_list[i]
        cplx2 = cplx_list[i+1]
        phase21 = estimate_phase_offset(freq1, freq2, cplx1, cplx2)
        phase_diff_list.append(phase21)
    if is_offset_equal:
        phase_offset = np.cumsum([np.mean(phase_diff_list)]*len(phase_diff_list))
    else:
        phase_offset = np.cumsum(phase_diff_list)
           
    freq_combined = freq_list[0]
    cplx_combined = cplx_list[0].copy()

    for i in range(1, len(freq_list)):
        freq_current = freq_list[i]
        cplx_current = cplx_list[i]*np.exp(-1j*phase_offset[i-1])
        idx_combined, idx_current = intersect_indices(freq_combined, freq_current)
        cplx_intersect = 0.5*(cplx_combined[idx_combined]+cplx_current[idx_current])
        cplx_combined[idx_combined] = cplx_intersect
        
        non_overlap_freq = np.setdiff1d(freq_current, freq_combined)
        freq_combined = np.concatenate((freq_combined, non_overlap_freq))
        cplx_combined = np.concatenate((cplx_combined, cplx_current[np.isin(freq_current, non_overlap_freq)]))
    return freq_combined, cplx_combined