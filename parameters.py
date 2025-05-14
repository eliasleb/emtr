import os

# Oscilloscope setup
TRIGGER_LEVEL = 200  # mV
TRIGGER_CHANNEL = 1
HORIZONTAL_SCALE = 500  # ns
HORIZONTAL_OFFSET = 5 * HORIZONTAL_SCALE  # ns
HORIZONTAL_OFFSET_TR = 10 * HORIZONTAL_SCALE
HORIZONTAL_OFFSET_ESD = 4 * HORIZONTAL_SCALE
OSC_UPPER_CUTOFF = 3  # GHz
SAMPLE_RATE = 80_000_000_000  # From 200_000 to 80_000_000_000, locked 1 2 4 8
N_ACTIVE_OSC_CHANNELS = 8
VERTICAL_SCALE_DIRECT_TIME = (200, ) * 8  # mV
VERTICAL_SCALE_TIME_REVERSAL = (20, ) * N_ACTIVE_OSC_CHANNELS  # mV
ALL_OSC_CHANNELS = set(range(1, 16 + 1))
AVERAGE_SWEEPS_ESD = 1

# TX setup
TX_CHANNELS = range(1, 8 + 1)

# Legacy
ARDUINO_PORT = ""


def active_osc_channels() -> tuple[int, ...]:
    """
    Convenience function to compute the active oscilloscope channels from the parameter :N_ACTIVE_OSC_CHANNELS:

    :return: A tuple of the range of active oscilloscope channels
    """
    return tuple(range(1, N_ACTIVE_OSC_CHANNELS + 1))
