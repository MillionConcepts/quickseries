"""lightweight version of `moonbow`'s polynomial fit functionality"""
from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import curve_fit


def fit_wrap(func, dimensionality, fit_parameters):
    @wraps(func)
    def wrapped_fit(independent_variable, *params):
        variable_components = [
            independent_variable[n] for n in range(dimensionality)
        ]
        exploded_function = func(*variable_components, *params)
        return exploded_function

    # rewrite the signature so that curve_fit will like it
    sig = signature(wrapped_fit)
    curve_fit_params = (
        Parameter("independent_variable", Parameter.POSITIONAL_ONLY),
        *fit_parameters,
    )
    wrapped_fit.__signature__ = sig.replace(parameters=curve_fit_params)
    return wrapped_fit


def fit(
    func: Callable,
    dimensionality: int,
    points: list[np.ndarray],
    dependent_variable: np.ndarray,
    guess: Optional[Sequence[float]] = None,
):
    sig = signature(func)
    assert dimensionality < len(sig.parameters), (
        "The model function must have at least one 'free' "
        "parameter to be a meaningful candidate for fitting."
    )
    fit_parameters = [
        item
        for ix, item in enumerate(sig.parameters.values())
        if ix >= dimensionality
    ]
    # TODO: check dim of dependent
    if not all(p.ndim == 1 for p in points):
        raise ValueError("each input vector must be 1-dimensional")
    if len(points) != dimensionality:
        raise ValueError(
            "number of input vectors does not match non-free parameters"
        )
    # TODO: optional goodness-of-fit evaluation
    return curve_fit(
        fit_wrap(func, dimensionality, fit_parameters),
        points,
        dependent_variable,
        maxfev=20000,
        p0=guess,
    )
