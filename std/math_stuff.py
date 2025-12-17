import numpy as np
import std
import scipy
import inspect


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


def find_mu(values, num_peaks=1, min_width=2):
    params = {
        "width": min_width,
        "prominence": 1.1
    }
    peaks, props = scipy.signal.find_peaks(values, **params)
    # print(props)
    prom_sort = np.argsort(props['prominences'])
    peaks = list(reversed(peaks[prom_sort]))
    return peaks[0:num_peaks]
    # return peaks


def gaussian(x, amp, mu, sigma):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    return amp * np.exp(-temp_a / temp_b)


def double_gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2, const):
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2) + const


def fit_func(func, x_values, y_values, x_errors=None, y_errors=None, p0=None):
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
    odr_run = scipy.odr.ODR(data, model, beta0=p0, maxit=7500)
    odr_run.run()

    params = odr_run.output.beta
    std_devs = odr_run.output.sd_beta
    goodness =goodness_of_fit(y_values, func(x_values, *params))
    return params, (std_devs, goodness)
