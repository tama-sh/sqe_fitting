from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

def plot_Sparameter(freq: np.ndarray, cplx: np.ndarray, frequency_unit: str = None, title: str = None, figsize: Tuple[float, float] = (12, 6), fig=None, axes=None, color=None):
    """Plot S parameter
    
    Args:
        freq (np.ndarray): frequency
        cplx (np.ndarray): complex signal
        frequency_unit (str): unit of frequency shown in x axis. default is None
        title (str): title of figure
        figsize (Tuple): size of figure
        fig (matplotlib.figure.Figure): figure object
        axes (List[matplotlib.axes.Axes]): list of axis object
        color: color of marker
        
    Returns:
        matplotlib.figure.Figure: figure
        List[matplotlib.axes.Axes]: list of axes
    """
    
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
            fig.suptitle(title)

        ax_amp = fig.add_subplot(221)
        ax_ph = fig.add_subplot(223)
        ax_polar = fig.add_subplot(122, projection="polar")
    
        if frequency_unit:
            freq_label = f"Frequency ({frequency_unit})"
        else:
            freq_lqbel = "Frequency"
    
        ax_amp.set_xlabel(freq_label)
        ax_amp.set_ylabel("Amplitude")
        ax_amp.grid(1)
        
        ax_ph.set_xlabel(freq_label)
        ax_ph.set_ylabel("Phase (rad)")
        ax_ph.grid(1)
    else:
        ax_amp = axes[0]
        ax_ph = axes[1]
        ax_polar = axes[2]

    cmap = plt.get_cmap('tab10')
    if color is None:
        color = cmap(0)
    elif isinstance(color, int):
        color = cmap(color)
    
    ax_amp.plot(freq, np.abs(cplx), "o", color=color, fillstyle="none", markersize=5)
    ax_ph.plot(freq, np.angle(cplx), "o", color=color, fillstyle="none", markersize=5)
    ax_polar.plot(np.angle(cplx), np.abs(cplx), "o", color=color, fillstyle="none", markersize=5)
     
    axes = [ax_amp, ax_ph, ax_polar]
    return fig, axes