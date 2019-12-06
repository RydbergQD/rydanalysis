import numpy as np


def get_calibration(ions, DPON=1, InitON=0, Roff=1):
    data = ions[
        (ions.DPON == DPON)
        & (ions.InitON == InitON)
        & (ions.Roff == Roff)
    ].ions
    return data.mean(), data.sem()


def plot_calibration(phase_scan, value, error, ax, color='tab:orange', label=''):
    value = value * np.ones(len(phase_scan))
    error = error * np.ones(len(phase_scan))
    ax.plot(
        phase_scan.index,
        value,
        color=color,
        label=label
    )
    ax.fill_between(
        x = phase_scan.index,
        y1 = value - error,
        y2 = value + error,
        color = color,
        alpha=0.3
    )