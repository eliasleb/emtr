import numpy as np
import matplotlib
import helpers
import emtr_experiment
from parameters import N_ACTIVE_OSC_CHANNELS, TX_CHANNELS
import matplotlib.pyplot as plt
import warnings



def esd_waveform(t):
    tau1, tau2, tau3, tau4 = 1.1e-9, 2e-9, 12e-9, 37e-9
    i1, i2 = 16.6, 9.3
    n = 1.8
    k1 = np.exp(-tau1/tau2 * (n * tau2/tau1)**(1/n))
    k2 = np.exp(-tau3/tau4 * (n * tau4/tau3)**(1/n))
    y = None
    with warnings.catch_warnings():
        y = i1 / k1 * (t/tau1)**n / (1 + (t/tau1)**n) * np.exp(-t/tau2) + \
            i2 / k2 * (t/tau3)**n / (1 + (t/tau3)**n) * np.exp(-t/tau4)
    y[np.isnan(y)] = 0.
    return y


def read_emtr_data(number, frequency, bw, esd_excitation=False, plot=True):
    n_tx_channels = len(TX_CHANNELS)
    data = helpers.read_calibration_data(number)
    t, f, tf, x_fft, keep, f_kept, channels, source, source_fd = None, None, None, None, None, None, None, None, None
    calibration = []
    if plot:
        plt.figure()
    for tx_ind, (tx_channel, d) in enumerate(data.items()):
        if not esd_excitation:
            d.filter(frequency, bandwidth=bw)
        calibration.append(np.sum(d.channels**2)**.5)
        if t is None:
            t = d.time
            f = d.freq
            source = emtr_experiment.source(t, frequency=frequency, bandwidth=bw)
            if esd_excitation:
                source = esd_waveform(t)
            tf = np.zeros((n_tx_channels, N_ACTIVE_OSC_CHANNELS, f.size), dtype="complex")
            channels = np.zeros((8, N_ACTIVE_OSC_CHANNELS, f.size))
        for rx_ind, rx_channel in enumerate(range(1, N_ACTIVE_OSC_CHANNELS + 1)):
            channels[tx_ind, rx_ind, :] = np.interp(
                t,
                d.time,
                d.channels[rx_ind, :]
            )
            print(f"Reading {tx_channel=}, {rx_channel=}")
            if esd_excitation:
                frequency_for_tf = None
                bw_for_tf = None
            else:
                frequency_for_tf = frequency
                bw_for_tf = bw
            tf_i, _ = helpers.get_transfer_function(t, source, channels[tx_ind, rx_ind, :], frequency_for_tf, bw_for_tf)
            tf[tx_ind, rx_ind, :] = tf_i
            if plot:
                plt.plot(f, np.abs(tf[tx_ind, rx_ind]))
    if plot:
        plt.xlim(frequency - bw, frequency + bw)
        plt.tight_layout()

    return t, f, tf, channels


def quality_of_convergence(true_channel, energy_tr):
    energy_tr = energy_tr / np.max(energy_tr)
    argmax = np.argmax(energy_tr)
    if argmax != true_channel - 1:
        return 0
    optimum = energy_tr[argmax]
    energy_tr[argmax] = 0.
    return optimum - np.max(energy_tr)


def main():
    pass

if __name__ == "__main__":
    matplotlib.use("TkAgg")

    main()
