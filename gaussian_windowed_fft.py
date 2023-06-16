import numpy as np
from scipy import fftpack
from scipy.signal import get_window, find_peaks
import matplotlib.pyplot as plt

def fourier_transform(time, signal, gauss_window = None):

    frequency = np.linspace(0,1/(2*(time[1]-time[0])), len(time)//2)
    freq_resolution = frequency[1]-frequency[0]

    time_len = time[-1] - time[0]

    relative_window_std_dev = gauss_window / time_len

    window_std_dev = relative_window_std_dev * np.size(signal)

    show = True # True for debug

    if window_std_dev == None:
        fourier_signal = np.abs(fftpack.fft(signal))
    
    else:
        if show:
            plt.plot(time, signal, marker = 'x')

        window = get_window(('gaussian', window_std_dev), np.size(signal))

        ## test rolled window
        window = get_window(('gaussian', window_std_dev), np.size(signal)*2)
        window = np.roll(window, np.size(signal))[:np.size(signal)]
        #plt.plot(time, window, marker = 'x')
        #plt.show()
        signal = signal * window
        
        if show:
    
            plt.plot(time, signal, marker = 'x')
            plt.show()

        fourier_signal = np.abs(fftpack.fft(signal))
    
    fourier_signal = fourier_signal[:len(time)//2]
    
    if show:
        plt.plot(frequency, fourier_signal, marker = 'x')
        plt.show()
    
    return(frequency, fourier_signal)

def find_peak(frequency, signal, prom = 8):
    
    try:
        peaks, _ = find_peaks(signal, prominence=prom)

        if len(peaks)==0:
            peak_index = None
            peak_frequency = None
            left_lobe_signal = None
            right_lobe_signal = None
            peak_signal = None

        if len(peaks)==1:
            peak_index = peaks[0]
            peak_frequency = frequency[peak_index]
            peak_signal = signal[peak_index]
            right_lobe_signal = signal[peak_index+1]
            left_lobe_signal = signal[peak_index-1]	

        if len(peaks)>1:
            signals = signal[peaks]
            max_peak = np.max(signals)
            peak_index = np.argwhere(signal==max_peak)
            peak_index = peak_index[0,0]
            peak_frequency = frequency[peak_index]
            peak_signal = signal[peak_index]
            right_lobe_signal = signal[peak_index+1]
            left_lobe_signal = signal[peak_index-1]	
            
            signals = signals/max_peak

            signals = np.delete(signals, np.argwhere(signals==1))
            if np.any(signals>0.33):
                peak_index = None
                peak_frequency = None
                left_lobe_signal = None
                right_lobe_signal = None
                peak_signal = None

    except:
        peak_index = None
        peak_frequency = None
        left_lobe_signal = None
        right_lobe_signal = None
        peak_signal = None
        
    #plt.plot(frequency, signal)
    #plt.plot(frequency[peak_index], signal[peak_index], 'o', color = 'g')

    #plt.plot(frequency[peaks], signal[peaks], 'x', color = 'r')	
    #plt.show()

    return(peak_frequency, peak_signal, left_lobe_signal, right_lobe_signal, peak_index)

def gaussian_estimation(peak_sig, left_lobe_sig, right_lobe_sig, peak_index, freq_res, f_min):
    p_sig = peak_sig 
    l_sig = left_lobe_sig
    r_sig = right_lobe_sig
    
    # print('p_sig:', p_sig)
    # print('l_sig:', l_sig)
    # print('r_sig:', r_sig)

    if p_sig == None:
        estimated_frequency = np.nan
    else:
        delta = np.log(r_sig/l_sig)/(2*np.log(p_sig**2/(l_sig*r_sig)))
        #print('hi:' + str(delta))
        estimated_frequency = f_min+freq_res*(peak_index+delta)
    
    return(estimated_frequency)

def Ramsey_freq_from_fft_interp(time, signal, gauss_window):
    
    frequency, fourier_signal = fourier_transform(time, signal, gauss_window = gauss_window)

    freq_res = frequency[1] - frequency[0]

    peak_frequency, peak_signal, left_lobe_signal, right_lobe_signal, peak_index = find_peak(frequency, fourier_signal)

    # plt.plot(frequency, fourier_signal, marker = 'x')
    # plt.show()
    
    estimated_frequency = gaussian_estimation(peak_signal, left_lobe_signal, right_lobe_signal, peak_index, freq_res, frequency[0])

    return estimated_frequency