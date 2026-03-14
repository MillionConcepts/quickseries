from pathlib import Path
import shutil
import time
import timeit

import numpy as np
from scipy.special import gamma

from quickseries import quickseries
from quickseries.sputils import lambdify

RNG = np.random.default_rng()

# TODO: make this less repetitive


def test_quickseries_1():
    kwargs = {
        'func': 'sin(x) * ln(x + cos(x))',
        'bounds': (0.5, 3.5),
        'nterms': 9,
        'cache': False
    }
    quick = quickseries(**kwargs)
    quick_nofit = quickseries(**kwargs, fit_series_expansion=False)
    lamb = lambdify(kwargs['func'])
    ax = np.arange(*kwargs['bounds'], 0.0005)
    lambda_value = lamb(ax)
    lambda_time = timeit.timeit(lambda: lamb(ax), number=50)
    quick_value = quick(ax)
    quick_time = timeit.timeit(lambda: quick(ax), number=50)
    quick_nofit_value = quick_nofit(ax)
    maxerr = abs(quick_value - lambda_value).max()
    maxerr_nofit = abs(quick_nofit_value - lambda_value).max()
    assert maxerr < 3e-4
    assert maxerr_nofit < 3e-4

    meanerr = abs(quick_value - lambda_value).mean()
    meanerr_nofit = abs(quick_nofit_value - lambda_value).mean()
    assert meanerr < 1e-4
    assert meanerr_nofit < 1e-4
    assert meanerr < meanerr_nofit
    assert quick_time < lambda_time


def test_quickseries_2():
    kwargs = {
        'func': 'cosh(x ** 3 * sin(y)) + sinh(y ** 3 * cos(x))',
        'bounds': (0, 1),
        'nterms': 6,
        'cache': False
    }
    quick = quickseries(**kwargs)
    lamb = lambdify(kwargs['func'])
    ax0 = np.arange(*kwargs['bounds'], 0.0005)
    ax1 = ax0.copy()
    RNG.shuffle(ax0)
    RNG.shuffle(ax1)
    lambda_value = lamb(ax0, ax1)
    lambda_time = timeit.timeit(lambda: lamb(ax0, ax1), number=50)
    quick_value = quick(ax0, ax1)
    quick_time = timeit.timeit(lambda: quick(ax0, ax1), number=50)
    maxerr = abs(quick_value - lambda_value).max()
    assert maxerr < 0.01
    assert quick_time < lambda_time


def test_quickseries_3():
    kwargs = {
        'func': '1 / gamma(x)',
        'bounds': (0, 1),
        'nterms': 7,
        'cache': False
    }
    ax0 = np.arange(*kwargs['bounds'], 0.0005)
    value = 1 / gamma(ax0)
    base_time = timeit.timeit(lambda: 1 / gamma(ax0), number=50)

    for basis in ("legendre", "chebyshev"):
        quick = quickseries(**kwargs, basis=basis)
        quick_value = quick(ax0)
        quick_time = timeit.timeit(lambda: quick(ax0), number=50)
        maxerr = abs(quick_value - value).max()
        assert quick_time < base_time
        assert maxerr < 1e-5


def test_approx_poly():
    poly = "x ** 3 * y + x ** 2 * y + 2 * y ** 2 + x + 3 * y + 4"
    poly_lambda = lambdify(poly)
    poly_quick = quickseries(poly)
    ax0 = np.arange(-10, 10, 0.0005)
    ax1 = ax0.copy()
    RNG.shuffle(ax0)
    RNG.shuffle(ax1)
    lambda_value = poly_lambda(ax0, ax1)
    lambda_time = timeit.timeit(lambda: poly_lambda(ax0, ax1), number=50)
    quick_value = poly_quick(ax0, ax1)
    quick_time = timeit.timeit(lambda: poly_quick(ax0, ax1), number=50)
    assert np.allclose(quick_value, lambda_value)
    assert quick_time < lambda_time


def test_jit():
    kwargs = {
        'func': 'sin(cos(sin(cos(sin(x) * x) * x) * x))',
        'bounds': (-3.14, 3.14),
        'nterms': 6,
        'cache': False
    }
    quick = quickseries(**kwargs)
    quick_jit = quickseries(**kwargs, jit=True)
    ax = np.arange(*kwargs['bounds'], 0.01)
    quick_jit(ax)  # trigger JIT compilation
    quick_time = timeit.timeit(lambda: quick(ax), number=50)
    quick_jit_time = timeit.timeit(lambda: quick_jit(ax), number=50)
    assert quick_jit_time < quick_time


def test_cache():
    shutil.rmtree(Path(__file__).parent / "__pycache__", ignore_errors=True)
    kwargs = {
        'func': 'sin(x ** 2) * cos(y ** 2) / (x + 10)',
        'bounds': (-3.14, 3.14),
        'nterms': 6,
        'cache': True
    }
    quickstart = time.time()
    quickseries(**kwargs)
    quicktime = time.time() - quickstart
    cachestart = time.time()
    quickseries(**kwargs)
    cachetime = time.time() - cachestart
    assert cachetime < quicktime
