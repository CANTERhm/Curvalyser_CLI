#!/usr/bin/python
# -*- coding: utf-8 -*-

# fit_range = (500, 3000)   # Arrow TL-2 (1st peak, in water)
fit_range = (4000, 10000)  # Arrow TL-2 (2nd peak, in water)
# fit_range = (1000, 8000)  # PNP-TR-TL, big lever (1st peak, in water)
# fit_range = (8000, 14000) # PNP-TR-TL, big lever (2nd peak, in water)
fit_range = None
resonance_correction = 2  # 0 = no correction, 1 = 1st peak, 2 = 2nd peak, ...
T = 28
# corrections = [0.81744, 0.29054]
corrections = [0.81744, 0.25107, 0.08631, 0.04412, 0.02669]
init_params = [1e-13, 1e3, 5, 0]

import math
import sys

import matplotlib.pyplot as pl
import numpy as np

SHO_func = lambda f, A, f0, Q, eta: A ** 2 * f0 ** 4 / (
(f ** 2 - f0 ** 2) ** 2 + f0 ** 2 * f ** 2 / Q ** 2) + eta ** 2  # simple harmonic oscillator


def _error_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def curve_fit(f, xdata, ydata, p0=None, **kw):
    from scipy.optimize import leastsq
    if p0 is None or np.isscalar(p0):
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(
            args) < 2: raise ValueError, 'p0 not given as a sequence and inspection cannot determine the number of fit parameters'
        if p0 is None: p0 = 1.0
        p0 = [p0] * (len(args) - 1)
    args = (xdata, ydata, f)
    popt, cov, infodict, msg, ier = leastsq(_error_function, p0, args=args, full_output=1, **kw)
    return (not ier), popt, (_error_function(popt, *args) ** 2).sum(), infodict['nfev'], msg


f, A = np.loadtxt(sys.argv[1], unpack=True)
f = f[20:]
A = A[20:]
if fit_range is None:
    start = 0
    stop = None
else:
    if fit_range[0] is None:
        start = 0
    else:
        for start in range(len(f)):
            if f[start] >= fit_range[0]: break
    if fit_range[1] is None:
        stop = None
    else:
        for stop in range(start, len(f)):
            if f[stop] > fit_range[1]: break
    init_params[1] = (fit_range[0] + fit_range[1]) / 2
    fit_error, popt, SS, nfev, msg = curve_fit(SHO_func, f[start:stop], A[start:stop], init_params)
    P = math.pi / 2 * popt[1] * popt[2] * popt[0] ** 2
    k_N = 1.3806504e-23 * (273.15 + T) / P
    if resonance_correction: print 'spring constant (corrected):   %8.3f mN/m' % (
    k_N * corrections[resonance_correction - 1] * 1e3)
    print 'spring constant (uncorrected): %8.3f mN/m' % (k_N * 1e3)
pl.plot(f, A)
pl.plot(f[start:stop], A[start:stop], 'k')
if fit_range is not None: pl.plot(f, SHO_func(f, *popt), lw=2)
pl.show()
