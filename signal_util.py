import numpy as np
import scipy.signal as scisig

def smoothing(data: np.ndarray, dt: float = 1, numtaps: int = 21, smoothing_width: float = 10):
    """Smooth the data by applying fir filter with zero phase.
    Args:
        data (np.ndarray): data to be smoothened
        dt (float): interval of sampling
        numtaps (int): number of taps used for fir filter
        smoothing_width (float): time width of smoothing, cutoff = 1/smoothing_width
        
    Returns:
        np.ndarray: smoothened data
    """
    cutoff = 1/smoothing_width
    b = scisig.firwin(numtaps=21, cutoff=cutoff, fs=dt)
    data_lp = scisig.filtfilt(b, 1, data) # apply zero phase filter
    return data_lp

def group_delay(omega: np.ndarray, cplx: np.ndarray):
    """Calculate group delay
    
    Use derivative of cplx instead of unwrapped phase for stability
    
    Args:
        omega (np.ndarray): angular frequency
        cplx (np.ndarray): complex data with electrical delay
        
    Returns:
        np.ndarray: group delay at (omega[1:]-omega[:-1])/2 ,the length of array become len(cplx)-1
    """
    return -np.imag((cplx[1:]-cplx[:-1])/(0.5*(cplx[1:]+cplx[:-1])))/(omega[1:]-omega[:-1])