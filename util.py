import numpy as np

def percentile_range_indices(data: np.ndarray, r_s: float, r_e: float):
    """Return indices of 1d array in a range of percentile 
    
    Args:
        data (np.ndarray): data
        r_s (float): ratio of data extracted from (0 to 1)
        r_e (float): ratio of data extracted to (0 to 1), must be greater than r_s
        
    Return:
        np.array: indices of array with the data of the given range of percentile
    """
    l = len(data)
    sort_idxs = np.argsort(data)
    idx_s = int(np.round(l*r_s))
    idx_e = int(np.round(l*r_e))
    
    idx_range = np.arange(idx_s, idx_e, 1)
    return sort_idxs[idx_range]

def percentile_range_data(data: np.ndarray, r_s: float, r_e: float):
    """Return indices of 1d array in a range of percentile 
    
    Args:
        data (np.ndarray): data
        r_s (float): ratio of data extracted from (0 to 1)
        r_e (float): ratio of data extracted to (0 to 1), must be greater than r_s
        
    Return:
        np.array: sorted data in the given range of percentile
    """
    return data[percentile_range_indices(data, r_s, r_e)]