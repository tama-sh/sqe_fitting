from typing import Tuple
import numpy as np

def intersect_indices(array1, array2):
    intersection = np.intersect1d(array1, array2)
    indices_array1 = np.where(np.isin(array1, intersection))[0]
    indices_array2 = np.where(np.isin(array2, intersection))[0]
    return indices_array1, indices_array2

def percentile_range_indices(data: np.ndarray, percentile_rage: Tuple[float, float]):
    """Return indices of 1d array in a range of percentile 
    
    Args:
        data (np.ndarray): data
        percentile_range: range of percentile to be extracted (from 0 to 1)
        
    Return:
        np.array: indices of array with the data of the given range of percentile
    """
    l = len(data)
    sort_idxs = np.argsort(data)
    idx_s = int(np.round(l*percentile_rage[0]))
    idx_e = int(np.round(l*percentile_rage[1]))
    
    idx_range = np.arange(idx_s, idx_e, 1)
    return sort_idxs[idx_range]

def percentile_range_data(data: np.ndarray, percentile_rage: Tuple[float, float]):
    """Return indices of 1d array in a range of percentile 
    
    Args:
        data (np.ndarray): data
        percentile_range: range of percentile to be extracted (from 0 to 1)
        
    Return:
        np.array: sorted data in the given range of percentile
    """
    return data[percentile_range_indices(data, percentile_rage)]