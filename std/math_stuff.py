import numpy as np
import std
import scipy
import inspect
import iminuit
from iminuit import cost
import numba
import propeller as p


def add(a, b):
    return a + b


def reduced_chi_2(data, fit, params, sigma=None):
    dof = len(data) - len(params)
    var = np.sqrt(np.var(data)) if std.util.none(sigma) else sigma
    return sum(((fit - data) / var) ** 2) / dof


def goodness_of_fit(data, fit):
    rss = sum((data - fit) ** 2)
    tss = sum((data - np.average(data)) ** 2)
    return 1 - (rss / tss)


def res_sq(f, x, y):
    return sum((f(x) - y) ** 2)


def make_initial_guesses(f, x, y, argc):
    # first ensure area under curve is somewhat similar
    return []


@np.vectorize
@numba.njit
def gaussian(x, amp, mu, sigma):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    return np.abs(amp) * np.exp(-temp_a / temp_b)


@np.vectorize
@numba.njit
def gaussian_ug(x, amp, mu, sigma, ug):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    return np.abs(amp) * np.exp(-temp_a / temp_b) + ug


@np.vectorize
@numba.njit
def area_gaussian(x, area, mu, sigma):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    amp = area / (np.sqrt(2 * np.pi) * sigma)
    return np.abs(amp) * np.exp(-temp_a / temp_b)


@np.vectorize
@numba.njit
def area_gaussian_ug(x, area, mu, sigma, ug):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    amp = area / (np.sqrt(2 * np.pi) * sigma)
    return np.abs(amp) * np.exp(-temp_a / temp_b) + ug


@numba.njit
def linear(x, a, b):
    return a * x + b


@numba.njit
def quadratic(x, a, b, c):
    return a * x * x + b * x + c


@np.vectorize
@numba.njit
def lorentzian(x, amp, mu, gamma):
    denominator = (x ** 2 - mu ** 2) ** 2 + (gamma ** 2) * (mu ** 2)
    return amp / denominator

def make_n_gaussian(n):
    return lambda x, *args: sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def make_n_area_gaussian(n):
    return lambda x, *args: sum([std.area_gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def double_gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2, const):
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2) + const


def lorentz_curve(x, a, x0, gamma):
    return a / ((x ** 2 - x0 ** 2) ** 2 + (gamma * x0) ** 2)


# def fit_func(func, x_values, y_values, x_errors=None, y_errors=None, p0=None, force_cf=False):
#     if std.none(p0):
#         argc = len(str(inspect.signature(func)).split()[1:])
#         p0 = np.ones(argc)

#     if std.none(y_errors):
#         y_errors = np.var(y_values)
#     cost_func = cost.LeastSquares(x_values, y_values, y_errors, func, loss="soft_l1")
#     m = iminuit.Minuit(cost_func, *p0)
#     m.migrad()
#     m.hesse()
#     goodness = goodness_of_fit(y_values, func(x_values, *m.values))
#     return m.values, (m.errors, goodness)

def odr_fit(func, x, y, p0=None, maxfev=1000):
    x_values, x_errors = p.ve(x) if isinstance(x[0], p.GenericOp) else (x, None)
    y_values, y_errors = p.ve(y) if isinstance(y[0], p.GenericOp) else (y, None)

    if std.some(x_errors):
        x_errors[x_errors == 0] = np.nan
    if std.some(y_errors):
        y_errors[y_errors == 0] = np.nan
    data = scipy.odr.RealData(x_values, y_values, x_errors, y_errors)
    argc = len(str(inspect.signature(func)).split()[1:])
    if std.none(p0):
        p0 = np.zeros(argc)
    func = np.vectorize(func)
    model = scipy.odr.Model(lambda B, t: func(t, *B[:argc]))
    odr_run = scipy.odr.ODR(data, model, beta0=p0, maxit=maxfev)
    odr_run.run()

    params_odr = odr_run.output.beta
    std_devs_odr = odr_run.output.sd_beta
    goodness_odr = goodness_of_fit(y_values, func(x_values, *params_odr))

    return params_odr, (std_devs_odr, goodness_odr)


def curve_fit(func, x, y, p0=None, maxfev=1000):
    x_values, x_errors = p.ve(x) if isinstance(x[0], p.GenericOp) else (x, None)
    y_values, y_errors = p.ve(y) if isinstance(y[0], p.GenericOp) else (y, None)

    argc = len(str(inspect.signature(func)).split()[1:])
    if std.none(p0):
        p0 = np.ones(argc)

    func = np.vectorize(func)
    params_cf, cov = scipy.optimize.curve_fit(func, x_values, y_values, sigma=y_errors, p0=p0, maxfev=maxfev)
    std_devs_cf = np.sqrt(np.diag(cov))
    goodness_cf = goodness_of_fit(y_values, func(x_values, *params_cf))
    return params_cf, (std_devs_cf, goodness_cf)


def fit_func(func, x_values, y_values, x_errors=None, y_errors=None, p0=None, force_cf=False, maxfev=99999):
    if force_cf:
        params_cf, cov = scipy.optimize.curve_fit(func, x_values,y_values, p0=p0, maxfev=maxfev)
        std_devs_cf = np.sqrt(np.diag(cov))
        goodness_cf = goodness_of_fit(y_values, func(x_values, *params_cf))
        return params_cf, (std_devs_cf, goodness_cf)

    try:
        model = scipy.odr.Model(lambda B, x: func(x, *B))
        if isinstance(x_errors, float):
            x_errors = np.ones(np.shape(x_values)) * x_errors
        if isinstance(y_errors, float):
            y_errors = np.ones(np.shape(y_values)) * y_errors
        if std.some(x_errors):
            x_errors[x_errors == 0] = np.nan
        if std.some(y_errors):
            y_errors[y_errors == 0] = np.nan
        data = scipy.odr.RealData(x_values, y_values, x_errors, y_errors)
        if std.none(p0):
            argc = len(str(inspect.signature(func)).split()[1:])
            p0 = np.zeros(argc)
            p0 += 1  # todo: put an estimator function for beta here
        odr_run = scipy.odr.ODR(data, model, beta0=p0, maxit=maxfev)
        odr_run.run()

        params_odr = odr_run.output.beta
        std_devs_odr = odr_run.output.sd_beta
        goodness_odr = goodness_of_fit(y_values, func(x_values, *params_odr))
    except Exception:
        return fit_func(func, x_values, y_values, x_errors, y_errors, p0, True)

    try:
        params_cf, cov = scipy.optimize.curve_fit(func, x_values,y_values, p0=p0, maxfev=maxfev)
        std_devs_cf = np.sqrt(np.diag(cov))
        goodness_cf = goodness_of_fit(y_values, func(x_values, *params_cf))
        if np.abs(goodness_cf - 1) < np.abs(goodness_odr - 1) or force_cf:
            return params_cf, (std_devs_cf, goodness_cf)
    except Exception:
        pass
    return params_odr, (std_devs_odr, goodness_odr)


def diff_find_maxima(y, smoothing=2, min_magnitude=0.):
    smoothing = 1 if smoothing < 1 else smoothing
    smooth_grad = np.gradient(np.convolve(y, np.ones(2 * smoothing), mode="same"))

    peaks = []

    before, after = np.zeros(len(y), dtype=np.bool), np.zeros(len(y), dtype=np.bool)
    for i in range(smoothing):
        before[i] = 1
        after[i + smoothing] = 1

    for i in range(smoothing, len(y) - smoothing):
        if np.all(smooth_grad[before] > min_magnitude) and np.all(smooth_grad[after] < min_magnitude):
            peaks.append(i - 1)
        before = np.roll(before, 1)
        after = np.roll(after, 1)

    return peaks


def diff_find_peaks(y, denoise=2, min_magnitude=0., min_sharpness=0., get_saddlepoints=False):
    denoise = 1 if denoise < 1 else denoise
    # smooth_grad = np.gradient(np.convolve(y, np.ones(2 * denoise), mode="same"))
    smooth_grad = np.gradient(y)

    peaks = []
    saddle_points = []

    for i in range(len(y)):
        if np.all(smooth_grad[i - denoise:i] > min_sharpness) and np.all(smooth_grad[i:i + denoise] < min_sharpness):
            peaks.append(i)
        elif np.all(smooth_grad[i - denoise:i] <= smooth_grad[i]) and np.all(smooth_grad[i + 1:i + denoise + 1] <= smooth_grad[i]) and np.abs(smooth_grad[i]) < 1 / min_sharpness:
            saddle_points.append(i)
        elif np.all(smooth_grad[i - denoise:i] >= smooth_grad[i]) and np.all(smooth_grad[i + 1:i + denoise + 1] >= smooth_grad[i]) and np.abs(smooth_grad[i]) < 1/ min_sharpness:
            saddle_points.append(i)

    if get_saddlepoints:
        print(saddle_points)
        return peaks + saddle_points
    return peaks


def find_mu(values, num_peaks=1, smoothing=2):
    peaks = diff_find_maxima(values, smoothing=smoothing)
    return peaks
    if len(peaks) == num_peaks:
        return peaks
    if len(peaks) < num_peaks and np.abs():
        return None
        

