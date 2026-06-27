error_bar_def = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}
import propeller as p
from matplotlib import pyplot as plt
import std

def plt_pretty(xlabel, ylabel):
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plt_finish(xlabel, ylabel, save_to=False):
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plt_errorbar(x, y, label=None):
    y_is_ev = isinstance(y[0], p.GenericOp)
    x_is_ev = isinstance(x[0], p.GenericOp)
    yval, yerr = p.ve(y) if y_is_ev else (y, None)
    xval, xerr = p.ve(x) if x_is_ev else (x, None)

    params = error_bar_def
    params["alpha"] = 0.75
    if std.none(xerr):
        params["fmt"] = "."
        params["markersize"] = 5
        

    plt.errorbar(xval, yval, xerr=xerr, yerr=yerr, label=label, **params)


def plt_func(f, params=None, label=None, xrange=None):
    import numpy as np
    xmin, xmax, _, _ = plt.axis()
    if std.some(xrange) and std.some(xrange[0]):
        xmin = xrange[0]
    if std.some(xrange) and std.some(xrange[1]):
        xmax = xrange[1]
    x = np.linspace(xmin, xmax, 10000)
    y = f(x) if std.none(params) else f(x, *params) 
    plt.plot(x, y, label=label)
