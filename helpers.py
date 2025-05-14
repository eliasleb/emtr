#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import warnings

import pandas as pd
import numpy as np
import scipy.interpolate
import re
from parameters import *
from dataclasses import dataclass
from typing import Optional
import serial
import time
from datetime import timedelta
import pickle
import hashlib
import unicodedata


class Switch:
    """Arduino-based switch control"""
    def __init__(self, logger: logging.Logger, serial_port: str = ARDUINO_PORT):
        try:
            self.arduino = serial.Serial(port=serial_port, baudrate=ARDUINO_BAUD_RATE, timeout=.1)
        except serial.serialutil.SerialException:
            import os 
            if os.name == "posix":
                logger.warning("Could not connect to Arduino")
            else:
                raise RuntimeError("Could not connect to Arduino")

        self.logger = logger

    def set_channel(self, channel: int):
        """
        Set a given channel by sending a serial command to the Arduino

        :param channel: New active channel
        """
        if channel not in TX_CHANNELS:
            raise ValueError(f"Channel {channel} not in available channels: {TX_CHANNELS}")

        channel_bytes = channel.to_bytes(1, "big")
        if self.arduino is not None:
            while self.arduino.read(size=1) != channel_bytes:
                self.arduino.write(channel_bytes)
                time.sleep(1e-6)
        self.logger.info(f"Activated switch channel {channel}")


def filter_data(fs: float, main_tone_frequency: float, bandwidth: float, y: np.ndarray):
    sos = scipy.signal.butter(
        2,
        [main_tone_frequency - bandwidth, main_tone_frequency + bandwidth],
        btype="bandpass",
        analog=False,
        output="sos",
        fs=fs
    )
    return scipy.signal.sosfiltfilt(sos, y)


@dataclass
class Data:
    """A class containing experiment data, with some basic signal processing features"""
    time: np.ndarray
    channels: np.ndarray
    _freq: Optional[np.ndarray] = None
    _fft: Optional[np.ndarray] = None
    i: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None

    def time_reverse(self):
        """Time-reverse the data by flipping the time axis and recomputing the FFT. The I and Q channels are reset."""
        self.channels = np.flip(self.channels, axis=1)
        self.do_fft()
        self.i = None
        self.q = None

    def __add__(self, other):
        if np.any(np.abs(self.time - other.time) > 1e-9):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(self.time, self.channels[0, :])
            plt.plot(other.time, other.channels[0, :])
            plt.show()

            raise NotImplementedError("Adding for different time bases not supported yet")
        self.channels = self.channels + other.channels
        if self._freq is not None:
            self._fft = self._fft + other.fft
            warnings.warn("Not sure about linearity of iq demod...")
            self.i = self.i + other.i
            self.q = self.q + other.q

        return self

    @property
    def fs(self):
        """Sampling frequency"""
        return 1 / self.dt

    @property
    def dt(self):
        """Sampling time"""
        if self.time.size < 2:
            raise RuntimeError("There must be at least two samples")
        return self.time[1] - self.time[0]

    @property
    def freq(self):
        """DFT frequencies"""
        if self._freq is None:
            self.do_fft()
        return self._freq

    @property
    def fft(self):
        """FFT of the channels"""
        if self._fft is None:
            self.do_fft()
        return self._fft

    def do_fft(self):
        """Computes the channel-wise FFT"""
        self._freq = np.linspace(0, self.fs, self.time.size)
        self._fft = np.fft.fft(self.channels)

    def down_sample(self, rate: int):
        """Down samples all channels and recomputes the FFT. May lead to aliasing."""
        self.time = self.time[::rate]
        self.channels = self.channels[:, ::rate]
        self.i = self.i[:, ::rate]
        self.q = self.q[:, ::rate]
        if self._freq is not None:
            self.do_fft()

    def filter(self, main_tone_frequency: float, bandwidth: float):
        """
        Perform a bandpass filter by setting frequencies outside the band to zero. Not the best in general, but seems to
        work OK in our case, if the bandwidth is large enough.

        :param main_tone_frequency: The center frequency of the bandpass.
        :param bandwidth: 1/2 the remaining bandwidth.
        """
        for id_channel, channel in enumerate(self.channels):
            self.channels[id_channel, :] = filter_data(self.fs, main_tone_frequency, bandwidth, channel)
        self.do_fft()

    def demodulate(self, fc: float):
        """
        Demodulate a high-frequency IQ-modulated signal.
        TODO Add support for phase correction

        :param fc: The estimated carrier frequency.
        """
        sos = scipy.signal.butter(2, fc, btype='low', analog=False, output='sos', fs=self.fs)
        self.i = self.channels.copy()
        self.q = self.channels.copy()
        for id_channel, channel in enumerate(self.channels):
            if channel[0] == np.nan:
                continue
            self.i[id_channel, :] = scipy.signal.sosfilt(
                sos,
                channel * np.cos(2 * np.pi * SDR_FREQUENCY * self.time)
            )
            self.q[id_channel, :] = scipy.signal.sosfilt(
                sos,
                channel * -1 * np.sin(2 * np.pi * SDR_FREQUENCY * self.time)
            )


def search_for_filenames(exp_number: int) -> list[str, ...]:
    """
    Goes through all log files in `./logs`, starting with the most recent one, and returns corresponding data filenames
     found in the log file. Kind of unstable, as it depends on the syntax of the log file. *Probable cause of issues.*

    :param exp_number: The C1 experiment number to look for.
    :return: A list with the found filenames.
    """
    log_files = os.listdir("logs")
    log_files.sort(reverse=True)
    for filename in log_files:
        try:
            with open(os.path.join("logs", filename), "r") as fd:
                text = " ".join(fd.readlines())
        except UnicodeDecodeError:
            with open(os.path.join("logs", filename), "r", encoding="latin1") as fd:
                text = " ".join(fd.readlines())
        expression = f"Experiment number: {exp_number}"
        if expression in text:
            before_that = text.split(expression)[0]
            before_that = before_that.split("INFO - Oscilloscope data: ")[-1]
            if f"C1--Trace--{exp_number:05}" in before_that:  # make sure we're talking about channel C1
                filenames = [name[1:-1] for name in re.findall(r"'.+?\.txt'", before_that) if "ConvertTo" not in name]
                return filenames

    return [os.path.join("data", "oscilloscope", f"C1--Trace--{exp_number:05}.txt"), ]


def read_local_data(filenames: list[str, ...], reading_tr_data=False) -> Data:
    """
    Tries to parse the data in `./data/oscilloscope` for all given filenames. Probable source of errors.

    :param filenames: A list of all data filenames.
    :param reading_tr_data:
    :return: A Data object whose `channels` property contains the data corresponding to :filenames:
    """
    data = None
    channels = None
    read_channels = set()
    for filename in filenames:
        kwargs = dict(skiprows=range(4))
        try:
            filename = filename.split("EMTR")[1][1:]
        except IndexError:
            pass  # Hopefully, we got a relative path
        try:
            temp = pd.read_csv(filename, **kwargs)
        except FileNotFoundError:
            filename = filename.replace("\\", "/").replace("//", "/").replace("C:", "/")
            if filename[0] == "/":
                filename = filename[1:]
            try:
                temp = pd.read_csv(filename, **kwargs)
            except FileNotFoundError:
                filename = filename.replace("data/", "data/oscilloscope/")
                temp = pd.read_csv(filename, **kwargs)

        channel = int(re.findall(r"C\d*", filename)[-1][1:])
        if channels is None:
            channels = np.nan * np.ones((len(filenames), len(temp)))
            t = np.asarray(temp["Time"])
            data = Data(
                time=t,
                channels=channels
            )
        try:
            if not reading_tr_data:
                data.channels[channel - 1] = np.interp(
                    data.time, temp.Time, temp.Ampl, left=0., right=0.
                )
            else:
                data.channels[0] = np.interp(
                    data.time, temp.Time, temp.Ampl, left=0., right=0.
                )
        except ValueError as e:
            # size_read, size_new = data.channels[0].size, temp.Ampl.size
            # if size_read == size_new + 1:
            #     data.channels[channel - 1][:size_new] = temp.Ampl
            # elif size_read == size_new - 1:
            #     data.channels[channel - 1] = temp.Ampl[:size_read]
            # else:
            print(f"Error while reading {filename}")
            raise e
        read_channels.add(channel)
        if set(active_osc_channels()).issubset(read_channels):
            break

    return data


def read_calibration_data(exp_number) -> dict[int, Data]:
    """Get all `Data` objects for the calibration whose first experiment number is :exp_number:
    :param exp_number: The experiment number of the first calibration (TX channel 1)
    :return: a dictionary mapping TX channels to the corresponding data
    """
    data = {}

    if type(exp_number) is int:
        filenames = search_for_filenames(exp_number)[:N_ACTIVE_OSC_CHANNELS]
        numbers = np.array([int(name[-8:-4]) for name in filenames])
        for tx_ind, tx in enumerate(TX_CHANNELS):
            files = [
                f.replace(f"{n:05d}", f"{n+tx_ind:05d}")
                for f, n in zip(filenames, numbers)
            ]
            data[tx] = read_local_data(files)
    else:
        _, channels, exp_numbers = parse_log_file(os.path.join("logs", f"{exp_number}.log"))
        if len(channels) != N_ACTIVE_OSC_CHANNELS * len(TX_CHANNELS):
            raise RuntimeError(
                f"There should be {N_ACTIVE_OSC_CHANNELS} x {len(TX_CHANNELS)} files for calibration {exp_number}")
        for tx, exp in zip(TX_CHANNELS, exp_numbers[::N_ACTIVE_OSC_CHANNELS]):
            filenames = search_for_filenames(exp)[:N_ACTIVE_OSC_CHANNELS]
            data[tx] = read_local_data(filenames)
    return data


class Interpolator(scipy.interpolate.interp1d):
    """Linear 1D interpolation, with constant extrapolation"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__doc__ = scipy.interpolate.interp1d.__doc__

    def __call__(self, *args, **kwargs) -> np.ndarray:
        x: np.ndarray = args[0].copy()
        idx: np.ndarray = (x < self.x.min()) | (x > self.x.max())

        x[x < self.x.min()] = self.x.min()
        x[x > self.x.max()] = self.x.max()

        y: np.ndarray = super().__call__(*((x, ) + args[1:]), **kwargs)
        y[idx] = 0
        return y


def txt_plot(z: np.ndarray, size: tuple[int, int], char="o") -> str:
    """
    Represents a 1D array as a coarse character-based plot. Since the array will be heavily down-sampled, this shows
    the cumulative energy, given by np.cumsum(z**2).

    :param z: Array to plot
    :param size: A tuple (width, height) such that the return string has :height: lines of :width: + 1 characters
    :param char: Fill-in character
    :return: A string containing the plot (only ' ', :char: and '\n')
    """
    if z.size > 1000:
        z = z[::z.size // 1000]
    z = np.cumsum(z**2)
    z = z / z[-1]
    x_z = np.linspace(0, size[0], z.size, endpoint=True)
    ret = ""
    for y in np.linspace(1, 0, size[1]):
        for x in range(size[0]):
            if y < z[np.where(x_z >= x)[0][0]]:
                ret += char[0]
            else:
                ret += " "
        ret += "\n"
    return ret


def test_txt_plot():
    t = np.linspace(0, 10)
    z = np.sin(t)

    print(txt_plot(z, (150, 8)))


def parse_log_file(filename: str):
    pattern = []
    data_contains_dt = True
    with open(filename, "r") as fd:
        while line := fd.readline():
            if "Oscilloscope data:" in line:
                pattern += re.findall(r"C(?P<channel>\d*)--Trace--(?P<exp>\d*).txt", line)
            if "Reading data from experiment #" in line:
                data_contains_dt = False
    channels = [int(s[0]) for s in pattern]
    exp_numbers = [int(s[1]) for s in pattern]
    return data_contains_dt, channels, exp_numbers


def get_transfer_function(
        t: np.ndarray, x: np.ndarray, y: np.ndarray, f0=None, bw=None,
        delete_below_ratio_of_max=0.01
) -> tuple[np.ndarray, ...]:
    fs = 1 / (t[1] - t[0])
    f = np.linspace(0, fs, t.size)
    x_fd, y_fd = np.fft.fft(x), np.fft.fft(y)
    # x_fd[np.abs(x_fd) < delete_below_ratio_of_max * np.max(np.abs(x_fd))] = np.nan
    # import matplotlib.pyplot as plt
    # plt.semilogy(f, np.abs(x_fd))
    # plt.show()
    tf = y_fd / x_fd
    if f0 is not None:
        valid_range = (f >= f0 - bw) & (f <= f0 + bw) | (f >= fs - f0 - bw) & (f <= fs - f0 + bw)
        tf[np.invert(valid_range)] = 0.
    tf[np.isnan(tf)] = 0.
    return tf, f


def setup_logger(environment, ):
    import sys
    logger = logging.getLogger(environment)
    logger.setLevel(logging.DEBUG)
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    timestamp = f'{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    file_handler = logging.FileHandler(filename=os.path.join("logs", f"{timestamp}.log"), mode='w')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format_string)
    for h in [file_handler, stream_handler]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger, timestamp


def manual_ctrl():
    sw = Switch(logger=logging.Logger("test"), serial_port="COM4")
    while True:
        for i in range(1, 9):
            print(i)
            sw.set_channel(i)
            time.sleep(2)


class EtaLogger:

    def __init__(self, n, logger, min_delay=.5):
        self.n = n
        self.logger = logger
        self._state = 0
        self._last_update = 0
        self._min_delay = min_delay

    def __enter__(self):
        now = time.time()
        self.start_time = now
        self._last_update = now
        return self

    def update(self):
        self._state += 1
        now = time.time()
        elapsed = now - self.start_time
        update_elapsed = now - self._last_update
        if update_elapsed >= self._min_delay:
            per_sec = self._state / elapsed
            remaining_updates = self.n - self._state
            eta = timedelta(seconds=remaining_updates / per_sec)
            self.logger.info(
                f"{self._state / self.n * 100:.2f} %, ETA: {eta} Elapsed: {timedelta(seconds=elapsed)} "
                f"({self._state} / {self.n}, {per_sec:.3f} / s)"
            )
            self._last_update = now

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class NoFwhm(Exception):
    pass


def fwhm(x, y):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the dataset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.max(y) / 2
    interp = scipy.interpolate.CubicSpline(x, y - half_max)
    roots = interp.roots()
    x_inter = np.linspace(np.min(x), np.max(x), x.size * 10)
    interpolated = interp(x_inter) + half_max
    roots = np.array(roots)

    roots[roots < np.min(x)] = np.nan
    roots[roots > np.max(x)] = np.nan
    roots = roots[np.invert(np.isnan(roots))]

    i_max = np.argmax(y)
    x_max = x[i_max]

    if len(roots) < 2:
        raise NoFwhm("Not enough roots")

    for i in range(roots.size - 1):
        if roots[i] <= x_max < roots[i+1]:
            return abs(roots[i] - roots[i+1]), x_inter, interpolated

    raise NoFwhm("No FWHM found")


def td_from_vna(f, x, axis=-1):
    df = f[1] - f[0]
    f_full = np.arange(0, 2 * np.max(f), df)
    i1 = np.where(f_full > f[0])[0][0]
    i2 = np.where(f_full > f[0])[0][-1]
    shape_before, shape_after = list(x.shape), list(x.shape)
    shape_before[axis] = i1
    shape_after[axis] = i2

    x_full = np.concatenate(
        (
            np.zeros(shape_before),
            x,
            np.zeros(shape_after),
            np.zeros(shape_after),
            np.conjugate(np.flip(x, axis=axis)),
            np.zeros(shape_before)
        ),
        axis=axis
    )
    f_full = np.linspace(0, df * x_full.shape[axis], x_full.shape[axis])
    t = np.linspace(0, 1 / df, f_full.size)
    return t, np.real(np.fft.ifft(x_full, axis=axis))


def fd_from_td_vna(t, x, f, axis=-1):
    fs = 1 / (t[1] - t[0])
    df = f[1] - f[0]
    f_full = np.linspace(0, fs, t.size)
    x_fd = np.fft.fft(x, axis=axis)
    return scipy.interpolate.interp1d(f_full, x_fd, axis=axis)(f + 2 * df)


def get_gaussian_fd(f_min, f_max, f):
    duration = 1 / (f_max - f_min) / 1
    f0 = (f_min + f_max) / 2
    return 1 / 2 * np.exp(-2 * (f - f0) ** 2 * (np.pi * duration) ** 2) * duration \
           + 1 / 2 * np.exp(-2 * (f + f0) ** 2 * (np.pi * duration) ** 2) * duration


def cache_function_call(func, *args, cache_dir="function_cache", logger=None, return_hash=False, **kwargs):
    """
    Caches the result of a function call and retrieves it from the cache if the same call is made again.

    Args:
    :func (function): The function to call.
    :*args: Positional arguments of the function.
    :cache_dir: Directory used to cache function calls
    :**kwargs: Keyword arguments of the function.

    Returns:
    The result of the function call.
    """
    msg = f"Calling function {func} with arguments {args} and kwargs {kwargs}"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Create a hash key from the function name and arguments
    args_hash = pickle.dumps((func.__name__, args, kwargs))
    hash_key = hashlib.sha256(args_hash).hexdigest()
    msg = f"Function call with hash {hash_key}"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    cache_path = os.path.join(cache_dir, hash_key + '.pkl')

    def handle_return(r):
        if not return_hash:
            return r
        if isinstance(r, tuple):
            return r + (hash_key, )
        return r, hash_key

    # Check if the result is cached
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        msg = "Retrieved from cache."
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        return handle_return(result)

    # Compute the result and cache it
    result = func(*args, **kwargs)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    return handle_return(result)


def sanitize_to_filename(s: str, max_length: int = 255) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', ascii_str)
    sanitized = re.sub(r'__+', '_', sanitized)
    sanitized = sanitized.strip('_-')

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_-')

    return sanitized or "file"


if __name__ == "__main__":
    # manual_ctrl()
    print(sanitize_to_filename("my_function(arg1, arg2)"))  # "my_function_arg1_arg2"
    print(sanitize_to_filename("Über-Cool😀Function!!!"))  # "Uber-Cool_Function"
    print(sanitize_to_filename(""))  # "file"
