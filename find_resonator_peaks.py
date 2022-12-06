#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from .util import correct_el_delay, sorted_partial_average
from scipy.signal import firwin

def find_resonator_peaks(freq, cplx, delay_threshold=None, min_distance=10, n_moving_average=11, with_plot=False):
    """
        Parameters:
            * freq: Frequency, 1D array of float (Unit xx)
            * cplx: Response from resonator, 1D array of complex
            * delay_threshold: Threshold of delay regarded as a resonator peak (Unit: 1/xx)
            * min_distance: Minimum distance between peaks (Unit xx)
    """
    if n_moving_average%2 == 0:
        freq_lp = (freq[int(n_moving_average/2)-1:-int(n_moving_average/2)] + freq[int(n_moving_average/2):int(-n_moving_average/2)+1])/2
    else:
        freq_lp = freq[int(n_moving_average/2):int(-n_moving_average/2)]

    cplx_lp = np.convolve(cplx, firwin(n_moving_average, 0.1), mode="valid")
    
    del_f = freq[1] - freq[0]
    cplx_c = correct_el_delay(freq_lp, cplx_lp)
    freq_diff = (freq_lp[1:]+freq_lp[:-1])/2
    delay = ((cplx_c[1:]-cplx_c[:-1])/(0.5*(cplx_c[1:]+cplx_c[:-1]))).imag/(2*np.pi*del_f)

    if delay_threshold is None:
        delay_threshold = 10*np.sqrt(sorted_partial_average(delay**2, 0.3, 0.7))
    peaks, _ = find_peaks(abs(delay), height=delay_threshold, distance=max(1, int(min_distance/del_f)))
    
    if with_plot:
        fig, ax = plt.subplots()
        ax.plot(freq_diff, delay)
        ax.plot(freq_diff[peaks], delay[peaks], "rx", markersize=10)
        ax.set_xlabel(f"Frequency")
        ax.set_ylabel(f"Group delay")
        fig.show()
    
    return freq_diff[peaks]