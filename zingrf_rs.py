import matplotlib.pyplot as plt
from helpers import *
import pickle
from matplotlib.backend_bases import MouseButton
from helpers import read_local_data, search_for_filenames
import itertools
from scipy.interpolate import RegularGridInterpolator
from emtr_fd import esd_waveform
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.animation as animation


if __name__ == "__main__":
    logger, timestamp = setup_logger(__name__)


def get_full_freq_mask(x, size):
    x_full = np.zeros((size, ))
    if 1 not in x:
        x[0] = 1
    block_size = x_full.size // len(x)
    for i, xi in enumerate(x):
        x_full[i * block_size:(i + 1) * block_size] = xi
    return x_full


def fitness_function(x, s12, s21, coeff=1., n_evaluate=10, plot=False):

    nx, ny = s12.shape[:2]
    error = 0.
    x_full = get_full_freq_mask(x, s12.shape[-1])
    calibration = np.sqrt(np.sum(np.abs(s12) ** 2 * x_full, axis=(2, 3)))
    i_s = np.random.randint(0, nx, size=n_evaluate)
    j_s = np.random.randint(0, ny, size=n_evaluate)
    for i, j in zip(i_s, j_s):
        tr = s12 * np.conjugate(s21[i, j, ...])[None, None, ...] * x_full
        tr = np.sum(tr, axis=2)
        energy = np.sqrt(np.sum(np.abs(tr) ** 2, axis=-1)) / calibration
        energy = energy / energy[i, j]
        energy[i, j] = 0.
        error += np.sum(energy ** 2)

        if plot:
            plt.clf()
            plt.contourf(energy.T)
            plt.plot(i, j, "wx", markersize=10)
            plt.pause(.5)

    error += coeff * np.sum(x)
    return (-error, )


def cluster_mutation(individual, cluster_size=5):
    """Mutates an individual by flipping small contiguous bit clusters."""
    start = random.randint(0, len(individual) - cluster_size)
    for i in range(start, start + cluster_size):
        individual[i] = 1 - individual[i]
    return individual,


def simulate_low_snr(f, s21, snr):
    t, _ = td_from_vna(f, s21[0, 0, 0, :], axis=-1)
    y_esd = esd_waveform(t)

    plt.figure()
    plt.plot(t, y_esd)
    plt.show(block=False)

    y_esd_fd = fd_from_td_vna(t, y_esd, f, axis=-1)
    s21 = s21 * y_esd_fd

    _, s21_td = td_from_vna(f, s21, axis=-1)

    rms = np.sqrt(np.mean(s21_td ** 2, axis=-1))
    noise = np.random.normal(0, 1, s21_td.shape)
    noise_rms = np.sqrt(np.mean(noise ** 2, axis=-1))
    s21_td = s21_td + noise / noise_rms[..., None] * rms[..., None] / snr

    plt.figure()
    plt.plot(t, s21_td[0, 0, 0, :])
    plt.show(block=False)

    s21_fd = fd_from_td_vna(t, s21_td, f, axis=-1)

    plt.figure()
    plt.plot(f, np.abs(s21[0, 0, 0, :]), label="original")
    plt.plot(f, np.abs(s21_fd[0, 0, 0, :]), label="noisy")
    plt.legend()
    plt.show(block=False)

    return s21_fd


def plot_generation(top_fitnesses, best, population, f1_megahertz=1000, f2_megahertz=3000):
    plt.subplot(3, 1, 1)
    plt.plot(top_fitnesses)
    plt.xlabel("Generation")
    plt.title("Elite imaging error")
    plt.ylabel("(1)")

    plt.subplot(3, 1, 2)
    x = np.linspace(f1_megahertz, f2_megahertz, len(best) + 1)
    plt.stairs(best, x)
    plt.title("Best individual")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("included")

    plt.subplot(3, 1, 3)
    plt.title("Population")
    plt.imshow(population)
    plt.xlabel("Frequency bin")
    plt.ylabel("Individual")
    plt.colorbar()
    plt.grid(False)

    plt.tight_layout()


@dataclass
class Individual:
    feature: list
    fitness: any

    def __iter__(self):
        return iter(self.feature)

    def __getitem__(self, index):
        return self.feature[index]

    def __setitem__(self, index, value):
        self.feature[index] = value

    def __len__(self):
        return len(self.feature)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


def select_frequencies_brute_force(
        f,
        s12, s21, snr=1e12,
        n_features=10,
        n_evaluate=23,
        n_cores=1,
        l1=0.
):
    if snr != np.inf:
        s21 = cache_function_call(simulate_low_snr, f, s21, snr)
    pop, fitnesses = [], []
    for x in itertools.product([0, 1], repeat=n_features):
        ind = Individual(list(x), np.nan)
        pop.append(ind)
    args = (s12, s21, l1, n_evaluate, )
    if n_cores != 1:
        results = Parallel(n_jobs=n_cores)(delayed(fitness_function)(ind, *args) for ind in pop)
        for ind, fit in zip(pop, results):
            ind.fitness = fit
    else:
        for ind in pop:
            ind.fitness = fitness_function(ind, *args)[0]

    sorted_pop = sorted(pop)
    best = sorted_pop[-1]
    x = get_full_freq_mask(best, f.size)

    s12 = s12 * x[None, None, None, :]
    s21 = s21 * x[None, None, None, :]
    return s12, s21, x


def read_data(filename, fun_logger=None):
    """"s11, s22, s12, s21"""
    path = f"data/vna/{filename}"
    try:
        with open(path + ".pickle", "rb") as fd:
            all_data = pickle.load(fd)
    except FileNotFoundError:
        pickle_files = [f for f in os.listdir(path) if f.endswith('.pickle')]
        pickle_files_sorted = sorted(pickle_files)
        all_data = []
        for trm_ind, f in enumerate(pickle_files_sorted):
            with open(os.path.join(path, f), "rb") as fd:
                data_list = pickle.load(fd)
                for i, element in enumerate(data_list):
                    element["trm_id"] = trm_ind
                    data_list[i] = element
                all_data += data_list

    trm_channels = sorted(list(set([d["trm_id"] for d in all_data])))
    trm_channel_to_index = {}
    for ind, trm_chan in enumerate(trm_channels):
        trm_channel_to_index[trm_chan] = ind
    n_trm = len(trm_channels)
    xs, ys, f = all_data[0]["xs"], all_data[0]["ys"], all_data[0]["f"]
    n_traces = next(iter(all_data))["data"].shape[0]
    data = np.zeros((xs.size, ys.size, n_trm, n_traces, f.size), dtype="complex")
    for d in all_data:
        data[d["i_x"], d["i_y"], trm_channel_to_index[d["trm_id"]], :, :] = d["data"]
    fun_logger.info(f"Number of RX antennas (TRM): {n_trm}; data shape: {data.shape}")
    return xs, ys, f, data


def read_data_from_td(exp_number, example_plot=False):
    filenames = search_for_filenames(exp_number)
    data = read_local_data(filenames=filenames)
    if example_plot:
        plt.plot(data.time * 1e6, data.channels[0, :] * 1e3, "k-")
        plt.xlim(-.3, 1.5)
        plt.ylabel("mV")
        plt.xlabel("Time (μs)")
        plt.title("Example time-domain data")
        plt.tight_layout()
        plt.show()
    data.do_fft()
    return data.freq, data.fft


def postprocessing(filename, filename_test=None,
                   learn_s11=None, td_exp_num=None, active_trace=1, f1=None, f2=None,
                   x0=None, y0=None, save=None, ml=False, do_montena_plots=False,
                   excite_gaussian=True, **fom_kwargs):
    """s_ij.shape: (x, y, rx, f)"""

    xs, ys, f, data = read_data(filename, fun_logger=logger)
    experimental_mismatch = filename_test is not None
    data_test, esd_waveform_fd = None, None
    n_traces = data.shape[3]
    if n_traces <= 4:
        ind_s11, ind_s22, ind_s12, ind_s21 = 0, 1, 2, 3
        ind_s33, ind_s44 = None, None
        ind_s23, ind_s24 = None, None
        ind_s13, ind_s14 = None, None
    else:
        ind_s11, ind_s22, ind_s12, ind_s21 = 0, 5, 1, 4
        ind_s33, ind_s44 = 10, 15
        ind_s23, ind_s24 = 6, 7
        ind_s13, ind_s14 = 2, 3
    if filename_test is not None:
        case = "mismatch"
    elif td_exp_num is not None:
        case = "td"
    else:
        case = "normal"
    s21_save_if_td = None
    if case == "mismatch":
        if filename_test != "self":
            _, _, _, data_test = read_data(filename_test, fun_logger=logger)
        if "zingrf_rs-2024-10-25-15-05-27_baseline_dut_4_mm_50_mm" == filename:
            # issue with trm channel 4
            data[:, :, 3, ...] = 0.
            data_test[:, :, 3, ...] = 0.
        # assert data_test.shape[:2] == (1, 1)
    elif case == "normal":
        s11, s22 = data[:, :, :, ind_s11, :], data[:, :, :, ind_s22, :]
        s12, s21 = data[:, :, :, ind_s12, :], data[:, :, :, ind_s21, :]
    elif case == "td":
        s11, s22 = data[:, :, :, ind_s11, :], data[:, :, :, ind_s22, :]
        s12, s21 = data[:, :, :, ind_s12, :], data[:, :, :, ind_s21, :]
        f_tf, tfs = read_data_from_td(td_exp_num, example_plot=do_montena_plots)
        assert tfs.shape[0] > 1

        s21_save_if_td = s21.copy()

        for ind in range(s11.shape[2]):
            s21[:, :, ind, :] = np.interp(f, f_tf, tfs[ind, :])[None, None, :]

        s21[np.isnan(s21)] = 0.
        t = np.linspace(0, 1 / (f[1] - f[0]), f.size)
        esd_waveform_fd = np.fft.fft(esd_waveform(t))

    if case == "mismatch":
        if filename_test != "self":
            s21_save_if_td = data[:, :, :, ind_s21, :].copy()
            s11_dut, s11 = data_test[:, :, :, ind_s11, :], data[:, :, :, ind_s11, :]
            s22, s22_dut = data[:, :, :, ind_s22, :], data_test[:, :, :, ind_s22, :]
            s12, s21 = data[:, :, :, ind_s12, :], data_test[:, :, :, ind_s21, :]
            if data_test.shape[:2] == (1, 1):
                s11_dut, s21 = np.tile(s11_dut, s22.shape[:2] + (1, 1, )), np.tile(s21, s12.shape[:2] + (1, 1, ))
                s22_dut = np.tile(s22_dut, s22.shape[:2] + (1, 1, ))
        else:
            if active_trace == 1:
                s11_dut, s11 = data[..., ind_s33, :], data[..., ind_s11, :]
                s22_dut, s22 = data[..., ind_s22, :], data[..., ind_s22, :]
                s12, s21 = data[..., ind_s12, :], data[..., ind_s23, :]
                if "tr_music" in fom_kwargs and fom_kwargs["tr_music"]:
                    s12 = s12 * data[..., ind_s13, :]
            elif active_trace == 2:
                s11_dut, s11 = data[..., ind_s44, :], data[..., ind_s11, :]
                s22_dut, s22 = data[..., ind_s22, :], data[..., ind_s22, :]
                s12, s21 = data[..., ind_s12, :], data[..., ind_s24, :]
                if "tr_music" in fom_kwargs and fom_kwargs["tr_music"]:
                    s12 = s12 * data[..., ind_s14, :]
            else:
                raise ValueError(f"Unknown {active_trace=}. Must be 1 or 2.")
    else:
        s11_dut, s22_dut = None, None

    true_delays, f_corrupt = None, None

    f1 = f1 if f1 is not None else np.min(f)
    f2 = f2 if f2 is not None else np.max(f)
    logger.info(f"Freq from {f.min()/1e6:.3f} MHz to {f.max()/1e6:.3f} MHz in {f.size} points.")
    logger.info(f"Case '{save}': f0 = {(f1 + f2) / 2e6:.3f} MHz, delta-f = {(f2 - f1) / 1e6:.3f} MHz")

    if ml == "brute_force":
        kwargs = dict(
            snr=1.,
            n_features=8,
            n_evaluate=15 * 15 // 2,
            n_cores=4,
        )
        start = time.perf_counter()
        s12, s21_modified, mask, hash_key = cache_function_call(
            select_frequencies_brute_force, f, s12,
            s21_save_if_td if td_exp_num is not None or filename_test is not None else s21,
            return_hash=True,
            **kwargs
        )
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.2f} seconds")
        plt.figure()
        plt.plot(f / 1e6, mask)
        plt.xlabel("Frequency (MHz)")
        plt.show(block=False)

        if td_exp_num is None and filename_test is None:
            s21 = s21_modified

    if do_montena_plots:
        montena_plots(f, s12)

    plot_resolution(f, xs, ys, s11, s11_dut, s12, s21, s22, s22_dut, f1, f2,
                    true_delays=true_delays, f_corrupt=f_corrupt, experimental_mismatch=experimental_mismatch,
                    learn_s11=learn_s11, excite_gaussian=excite_gaussian, _esd_waveform_fd=esd_waveform_fd,
                    x0=x0, y0=y0,
                    plot_filename=f"{filename}_test_{filename_test}_exp_number_{td_exp_num}" if save is None else save,
                    **fom_kwargs)
    return s11


def montena_plots(f, s12):
    plt.semilogy(f * 1e-9, np.abs(s12[0, 0, 0, :]), "k-")
    plt.xlim(0, 3)
    plt.xlabel("Frequency (GHz)")
    plt.title("Example scattering parameter")
    plt.tight_layout()
    plt.show(block=True)


def plot_resolution(f, xs, ys, s11, s11_dut, s12, s21, s22, s22_dut, f1, f2,
                    experimental_mismatch=False, learn_s11=None, excite_gaussian=True, _esd_waveform_fd=None,
                    plot_filename=None, x0=None, y0=None, manual_max=1., block_plot=False, auto_close=False,
                    **fom_kwargs):

    fom_kwargs.setdefault("animation_title", None)

    plt.figure(figsize=(10, 9))
    n_row, n_col = 2, 2

    ind_t = 0

    choosing_f1, recovered_phases = True, None
    xs_dense, ys_dense = np.linspace(np.min(xs), np.max(xs), 100), np.linspace(np.min(ys), np.max(ys), 100)
    dense_list_of_points = np.array([(x, y) for (x, y) in itertools.product(xs_dense, ys_dense)])
    x0 = np.min(xs) if x0 is None else x0
    y0 = np.min(ys) if y0 is None else y0

    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    dx_dense, dy_dense = xs_dense[1] - xs_dense[0], ys_dense[1] - ys_dense[0]
    removed_channels = set()
    fom_title = f"FOM, {plot_filename}"

    def plot_at():
        nonlocal x0, y0, recovered_phases, ind_t, xs_dense, ys_dense
        plt.clf()
        s12_corr, s21_corr = s12.copy(), s21.copy()
        factor = (f1 + f2) / 2 / 3e11 * 1e2

        xs_wavelengths, ys_wavelengths = xs_dense * factor, ys_dense * factor
        ind_x, ind_y = int(np.round((x0 - np.min(xs)) / dx)), int(np.round((y0 - np.min(ys)) / dy))
        ind_x, ind_y = np.clip(ind_x, 0, xs.size - 1), np.clip(ind_y, 0, ys.size - 1)
        ind_x_dense = int(np.round((x0 - np.min(xs_dense)) / dx_dense))
        ind_x_dense = np.clip(ind_x_dense, 0, xs_dense.size - 1)
        ind_y_dense = int(np.round((y0 - np.min(ys_dense)) / dy_dense))
        ind_y_dense = np.clip(ind_y_dense, 0, ys_dense.size - 1)

        for channel in removed_channels:
            s12_corr[:, :, channel - 1, ...] = 0.
            s21_corr[:, :, channel - 1, ...] = 0.

        if excite_gaussian:
            excitation_signal = get_gaussian_fd(f1, f2, f)
        elif _esd_waveform_fd is not None:
            excitation_signal = _esd_waveform_fd * get_gaussian_fd(f1, f2, f)
        else:
            excitation_signal = (f >= f1) * (f <= f2)

        fom, ratio = get_fom(
            f, s11, s11_dut, s12_corr, s21_corr, s22, s22_dut, ind_x, ind_y, f1, f2, _ind_t=ind_t,
            _experimental_mismatch=experimental_mismatch, _learn_s11=learn_s11,
            excitation_signal=excitation_signal, xs=xs, ys=ys, **fom_kwargs
        )
        try:
            fom_inter = RegularGridInterpolator((xs, ys), fom, method="cubic")(dense_list_of_points)
        except ValueError:
            fom_inter = fom
            xs_dense, ys_dense = xs, ys
            xs_wavelengths, ys_wavelengths = xs_dense * factor, ys_dense * factor
        fom_inter = fom_inter.reshape((xs_dense.size, ys_dense.size))
        fom_inter[fom_inter < 0.] = 0.

        fom_inter_clipped = np.clip(fom_inter, 0, manual_max)

        plt.subplot(n_row, n_col, 2)
        m = manual_max
        plt.contourf(xs_dense, ys_dense, fom_inter_clipped.T, cmap="jet", levels=np.linspace(0, m, 21))
        plt.colorbar(ticks=np.arange(0, 1.1, .25))
        plt.contour(xs, ys, fom.T, levels=(.5, ))
        plt.hlines(y0, xmin=np.min(xs), xmax=np.max(xs), color="w")
        plt.vlines(x0, ymin=np.min(ys), ymax=np.max(ys), color="w")

        plt.title(fom_title)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        plt.subplot(n_row, n_col, 4)
        plt.plot(xs_dense, fom_inter[:, ind_y_dense], "k-")
        plt.ylim(0, 1.1 * np.max(fom_inter[:, ind_y_dense]))
        plt.xlim(min(xs), max(xs))
        plt.hlines(.5 * np.max(fom_inter[:, ind_y_dense]), min(xs), max(xs), colors="r")
        try:
            full_width_half_max, x_inter, interpolated = fwhm(xs_wavelengths, fom_inter[:, ind_y_dense])
            plt.title(f"FWHM = λ / {100 / full_width_half_max:.2f}")

        except NoFwhm:
            pass
        plt.xlabel("x (mm)")

        plt.subplot(n_row, n_col, 1)
        plt.plot(fom_inter[ind_x_dense, :], ys_dense, "k-")
        plt.xlim(0, 1.1 * np.max(fom_inter[ind_x_dense, :]))
        plt.ylim(min(ys), max(ys))
        plt.vlines(.5 * np.max(fom_inter[ind_x_dense, :]), min(ys), max(ys), colors="r")
        try:
            full_width_half_max, x_inter, interpolated = fwhm(ys_wavelengths, fom_inter[ind_x_dense, :])
            plt.title(f"FWHM = λ / {100 / full_width_half_max:.2f}")
            # plt.plot(interpolated, x_inter / factor)
        except NoFwhm:
            pass
        plt.ylabel("y (mm)")

        plt.subplot(n_row, n_col, 3)
        trm_slice = slice(None)
        to_plot = np.abs(s12[ind_x, ind_y, trm_slice, :].reshape(-1, f.size)).T

        plt.semilogy(f, to_plot, "-")

        plt.text(
            f1 + .15 * (f2 - f2),
            np.max(np.abs(to_plot)) * .95,
            f"f1 = {f1/1e9:.3f} GHz, f2 = {f2/1e9:.3f} GHz"
        )

        plt.xlim(f1, f2)
        plt.title("Scattering parameters")
        plt.xlabel("Frequency (Hz)")

        plt.tight_layout()
        plt.gcf().canvas.draw_idle()

    plot_at()
    for extension in ("png", "pdf"):
        plt.savefig(f"figs/fom_{plot_filename}.{extension}")

    def on_click(event):
        nonlocal x0, y0, f1, f2, choosing_f1, ind_t
        if event.button is MouseButton.LEFT:
            if event.inaxes:
                if event.inaxes.title.get_text() == fom_title:
                    x0, y0 = event.xdata, event.ydata
                    x0, y0 = np.round(x0 / dx_dense) * dx_dense, np.round(y0 / dy_dense) * dy_dense
                    x0, y0 = min(np.max(xs), x0), min(np.max(ys), y0)
                    x0, y0 = max(np.min(xs), x0), max(np.min(ys), y0)
                elif event.inaxes.title.get_text() == "Scattering parameters":
                    chosen_f = event.xdata
                    if choosing_f1:
                        f1 = chosen_f
                        choosing_f1 = not choosing_f1
                    else:
                        if chosen_f > f1:
                            f2 = chosen_f
                            choosing_f1 = not choosing_f1
                    plt.subplot(2, 3, 4)
                    plt.plot([f1, f2], [0, 0], "r+", markersize=20)

                plot_at()
        if event.button is MouseButton.RIGHT:
            if event.inaxes.title.get_text() == "Scattering parameters":
                f1, f2 = np.min(f), np.max(f)
                choosing_f1 = True
                plot_at()
            elif event.inaxes.title.get_text() == "FOM":
                ind_t += 1
                plot_at()

    def on_key_press(event):
        nonlocal f1, f2, x0, y0
        df = (f2 - f1) / 2
        f0 = (f1 + f2) / 2
        df_smol = df / 10
        if event.key == "right":
            if f2 + df < np.max(f):
                f1, f2 = f1 + df / 2, f2 + df / 2
        elif event.key == "left":
            if f1 - df > np.min(f):
                f1, f2 = f1 - df / 2, f2 - df / 2
        elif event.key == "shift+right":
            if f2 + df_smol < np.max(f):
                f1, f2 = f1 + df_smol / 2, f2 + df_smol / 2
        elif event.key == "shift+left":
            if f1 - df_smol > np.min(f):
                f1, f2 = f1 - df_smol / 2, f2 - df_smol / 2
        elif event.key == "up":
            df /= 1.1
            f1 = f0 - df
            f2 = f0 + df
            f1, f2 = max(f1, np.min(f)), min(f2, np.max(f))
        elif event.key == "down":
            df *= 1.1
            f1 = f0 - df
            f2 = f0 + df
            f1, f2 = max(f1, np.min(f)), min(f2, np.max(f))
        elif event.key == "shift+up":
            df /= 1.01
            f1 = f0 - df
            f2 = f0 + df
            f1, f2 = max(f1, np.min(f)), min(f2, np.max(f))
        elif event.key == "shift+down":
            df *= 1.01
            f1 = f0 - df
            f2 = f0 + df
            f1, f2 = max(f1, np.min(f)), min(f2, np.max(f))

        elif event.key == "super+right":
            x0 += dx
            x0 = min(x0, np.max(xs))
        elif event.key == "super+left":
            x0 -= dx
            x0 = max(x0, np.min(xs))
        elif event.key == "super+up":
            y0 += y0
            y0 = min(y0, np.max(ys))
        elif event.key == "super+down":
            y0 -= dy
            y0 = max(y0, np.min(xs))
        elif event.key == "p":
            for file_extension in ("png", "pdf"):
                plt.savefig(f"figs/fom_{plot_filename}.{file_extension}")
        elif event.key == "1":
            res = input("f1 (MHz): ")
            f1 = int(res) * 1e6
        elif event.key == "2":
            res = input("f2 (MHz): ")
            f2 = int(res) * 1e6

        plot_at()

    plt.connect('button_press_event', on_click)
    plt.connect("key_press_event", on_key_press)
    plt.show(block=block_plot)
    if auto_close:
        plt.close()


def get_fom(f,
            _s11, _s11_dut, s12, s21, _s22, _s22_dut,
            ind_x, ind_y, _f1, _f2, time_domain=False, normalize=True, _experimental_mismatch=False, _ind_t=None,
            _learn_s11=None, excitation_signal=None, calibrate=False, min_normalization=False,
            xs=None, ys=None, animation_title=None, animation_length_ns=10):
    """s_ij.shape: (x, y, rx, f)"""

    s12, s21 = s12[None, ...], s21[None, ...]

    if calibrate:
        calibration = np.sum(np.abs(s12 * excitation_signal) ** 2, axis=(0, 3, 4)) ** .5
    else:
        calibration = None

    trm_measurement = s21[:, ind_x, ind_y, :, ...][:, None, None, ...] * excitation_signal

    tr = np.sum(
        s12 * np.conj(trm_measurement),
        axis=3
    )

    if time_domain:
        t, tr_td = td_from_vna(f, tr, axis=-1)
        dt = t[1] - t[0]
        if calibrate:
            tr_td = tr_td / calibration[:, :, None]
        m = np.max(np.abs(tr_td))
        ind_max = np.argmax(np.max(np.abs(tr_td), axis=(0, 1, 2)), axis=-1)
        logger.info(f"{t[ind_max]=}")
        if animation_title is not None:
            fig, ax = plt.subplots(figsize=(4.7, 4))
            cax = None
            levels = np.linspace(-1, 1, 51)
            n_indices = int(animation_length_ns * 1e-9 / dt)
            indices = range(ind_max - n_indices // 2, ind_max + n_indices // 2)
            logger.info(f"Number of frames: {len(indices)}")
            save_path = f"figs/tr_animation_{animation_title}.mp4"
            writer = animation.FFMpegWriter(fps=30)
            with writer.saving(fig, save_path, dpi=120):
                for ind_t in tqdm(indices, desc="Rendering frames"):
                    ax.clear()
                    if xs is not None:
                        args = xs, ys, tr_td[0, :, :, ind_t].T / m
                    else:
                        args = (tr_td[0, :, :, ind_t].T / m, )
                    cf = ax.contourf(*args, levels=levels, cmap="bwr")
                    t_i = t[ind_t]
                    if ind_t < 0:
                        t_i -= np.max(t)
                    if ind_t == ind_max:
                        marker = "*"
                    else:
                        marker = ""
                    ax.set_title(f"t = {t_i * 1e9:.04f} ns {marker}")
                    ax.set_xlabel("x (mm)")
                    ax.set_ylabel("y (mm)")
                    if cax is None:
                        cax = fig.colorbar(cf, ax=ax)
                        cax.set_ticks([xi / 10 for xi in range(-10, 11, 2)])
                    plt.tight_layout()
                    writer.grab_frame()
            plt.close(fig)

        ind_max = np.argmax(np.max(np.abs(tr_td), axis=(0, 1, 2)))
        fom = np.sum(tr_td[..., ind_max]**2, axis=0) ** .5
    else:
        if calibrate:
            fom = np.sum(np.abs(tr) ** 2, axis=(0, -1)) ** .5 / calibration
        else:
            fom = np.sum(np.abs(tr) ** 2, axis=(0, -1)) ** .5

    if min_normalization:
        fom -= np.min(fom)

    if normalize:
        return fom / np.max(fom), None
    return fom, None


def get_metrics(fom, x0, y0, xs, ys):
    ind_max = np.unravel_index(np.argmax(fom), fom.shape)
    x_guess, y_guess = xs[ind_max[0]], ys[ind_max[1]]
    error = np.sqrt((x0 - x_guess) ** 2 + (y0 - y_guess) ** 2)
    return error


def postprocessing_fourth_campaign():
    fom_kwargs = dict(
        calibrate=True,
        min_normalization=False,
        block_plot=True,
        auto_close=False,
        time_domain=False,
        animation_length_ns=1
    )
    kwargs = dict(
        excite_gaussian=True,
        f1=.77e9,
        f2=.9e9,
    )

    # RMLv5 validation
    postprocessing("zingrf_rs-2025-04-09-17-01-42_baseline",
                   x0=50, y0=50,
                   save="baseline_validation",
                   **kwargs, **fom_kwargs,)
    postprocessing("zingrf_rs-2025-04-09-10-08-29_rmlv5",
                   x0=50, y0=50,
                   save="rmlv5_validation",
                   **kwargs, **fom_kwargs,)

    # In general: ESD = 2 kV, shielded
    kwargs["f1"] = .35e9
    kwargs["f2"] = 1.8e9

    # Animation
    fom_kwargs["time_domain"] = True
    fom_kwargs["animation_title"] = "dut_rmlv5_bundled_esd_y1_for_td"
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   td_exp_num=4542,
                   **kwargs, **fom_kwargs)
    fom_kwargs["animation_title"] = "dut_rmlv5_bundled_esd_y2_for_td"
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   td_exp_num=4541,
                   **kwargs, **fom_kwargs)
    fom_kwargs["time_domain"] = False

    # DUT, RMLv3, cables bundled in a shield
    # Synth
    postprocessing("zingrf_rs-2025-04-17-11-53-33_dut_rmlv3",
                   filename_test="zingrf_rs-2025-04-17-13-36-25_dut_rmlv3_y1",
                   save="dut_rmlv3_bundled_synth_y1",
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-17-11-53-33_dut_rmlv3",
                   filename_test="zingrf_rs-2025-04-17-13-37-17_dut_rmlv3_y2",
                   save="dut_rmlv3_bundled_synth_y2",
                   **kwargs, **fom_kwargs)

    # ESD
    postprocessing("zingrf_rs-2025-04-17-11-53-33_dut_rmlv3", td_exp_num=4552,
                   save="dut_rmlv3_bundled_esd_y1",
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-17-11-53-33_dut_rmlv3", td_exp_num=4551,
                   save="dut_rmlv3_bundled_esd_y2",
                   **kwargs, **fom_kwargs)

    # DUT, cables bundled in a shield
    kwargs["f1"] = .2e9
    kwargs["f2"] = 2e9
    # Synth
    postprocessing("zingrf_rs-2025-04-15-11-23-41_dut_baseline",
                   save="dut_baseline_bundle_synth_y1",
                   filename_test="zingrf_rs-2025-04-15-13-10-14_dut_baseline_y1",
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-15-11-23-41_dut_baseline",
                   filename_test="zingrf_rs-2025-04-15-13-09-22_dut_baseline_y2",
                   save="dut_baseline_bundle_synth_y2",
                   **kwargs, **fom_kwargs)
    # ESD
    postprocessing("zingrf_rs-2025-04-15-11-23-41_dut_baseline",
                   save="dut_baseline_bundle_esd_y1",
                   td_exp_num=4538,
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-15-11-23-41_dut_baseline",
                   save="dut_baseline_bundle_esd_y2",
                   td_exp_num=4537,
                   **kwargs, **fom_kwargs)

    # DUT + RMLv5, cables bundled in a shield
    # Synth
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   filename_test="zingrf_rs-2025-04-15-16-05-47_dut_rmlv5_y1",
                   save="dut_rmlv5_bundled_synth_y1",
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   filename_test="zingrf_rs-2025-04-15-16-06-51_dut_rmlv5_y2",
                   save="dut_rmlv5_bundled_synth_y2",
                   **kwargs, **fom_kwargs)

    # ESD
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   save="dut_rmlv5_bundled_esd_y1",
                   td_exp_num=4542,
                   **kwargs, **fom_kwargs)
    postprocessing("zingrf_rs-2025-04-15-14-44-55_dut_rmlv5",
                   save="dut_rmlv5_bundled_esd_y2",
                   td_exp_num=4541,
                   **kwargs, **fom_kwargs)

    # No DUT
    for kwargs, name in zip(
            (dict(f1=2.5e9), dict(f1=.6e9, f2=1.e9, ), dict(f1=.2e9, f2=1.6e9, )),
            ("high", "low5", "low2", )
    ):
        # Baseline
        postprocessing("zingrf_rs-2025-04-09-17-01-42_baseline", td_exp_num=4527, **kwargs, x0=25, y0=25,
                       save=f"baseline_esd_{name}_25_25",
                       **fom_kwargs)  # 25 25 mm, 2 kV
        postprocessing("zingrf_rs-2025-04-09-17-01-42_baseline", td_exp_num=4528, **kwargs, x0=50, y0=50,
                       save=f"baseline_esd_{name}_50_50",
                       **fom_kwargs)  # 50 50 mm, 2 kV
        postprocessing("zingrf_rs-2025-04-09-17-01-42_baseline", td_exp_num=4529, **kwargs, x0=75, y0=75,
                       save=f"baseline_esd_{name}_75_75",
                       **fom_kwargs)  # 75 75 mm, 2 kV

        # RMLv5
        postprocessing("zingrf_rs-2025-04-09-10-08-29_rmlv5", td_exp_num=4524, x0=50, y0=50, **kwargs,
                       save=f"rmlv5_esd_{name}_50_50",
                       **fom_kwargs)  # 2 kV, 50 50 mm, shielded
        postprocessing("zingrf_rs-2025-04-09-10-08-29_rmlv5", td_exp_num=4525, x0=25, y0=25, **kwargs,
                       save=f"rmlv5_esd_{name}_25_25",
                       **fom_kwargs)  # as before, 25 25 mm
        postprocessing("zingrf_rs-2025-04-09-10-08-29_rmlv5", td_exp_num=4526, x0=75, y0=75, **kwargs,
                       save=f"rmlv5_esd_{name}_75_75",
                       **fom_kwargs)  # as before, 75 75 mm

        # RMLv3
        postprocessing("zingrf_rs-2025-04-16-15-57-02_rmlv3", td_exp_num=4544, x0=25, y0=25,
                       save=f"rmlv3_esd_{name}_25_25",
                       **kwargs, **fom_kwargs)  # 25 25
        postprocessing("zingrf_rs-2025-04-16-15-57-02_rmlv3", td_exp_num=4545, x0=50, y0=50,
                       save=f"rmlv3_esd_{name}_50_50",
                       **kwargs, **fom_kwargs)  # 50 50
        postprocessing("zingrf_rs-2025-04-16-15-57-02_rmlv3", td_exp_num=4548, x0=75, y0=75,
                       save=f"rmlv3_esd_{name}_75_75",
                       **kwargs, **fom_kwargs)  # 75 75


def main():
    postprocessing_fourth_campaign()


if __name__ == "__main__":
    import matplotlib
    import platform

    if platform.system() == "Windows":
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("MacOSX")

    plt.rcParams["font.family"] = "Times New Roman"

    plt.style.use("ggplot")
    plt.rcParams['pdf.fonttype'] = 42

    main()
