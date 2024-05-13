"""lightweight version of `moonbow`'s polynomial fit functionality"""
from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import curve_fit


def fit_wrap(
    func: Callable[[np.ndarray | float, ...], np.ndarray | float],
    dimensionality: int,
    fit_parameters: Sequence[str]
) -> Callable[[np.ndarray | float, ...], np.ndarray | float]:
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
    vecs: list[np.ndarray],
    dependent_variable: np.ndarray,
    guess: Optional[Sequence[float]] = None,
    bounds: Optional[
        Union[tuple[tuple[float, float]], tuple[float, float]]
    ] = None
) -> tuple[np.ndarray, np.ndarray]:
    sig = signature(func)
    assert len(vecs) < len(sig.parameters), (
        "The model function must have at least one 'free' "
        "parameter to be a meaningful candidate for fitting."
    )
    fit_parameters = [
        item
        for ix, item in enumerate(sig.parameters.values())
        if ix >= len(vecs)
    ]
    # TODO: check dim of dependent
    if not all(p.ndim == 1 for p in vecs):
        raise ValueError("each input vector must be 1-dimensional")
    # TODO: optional goodness-of-fit evaluation
    kw = {'bounds': bounds} if bounds is not None else {}
    # noinspection PyTypeChecker
    return curve_fit(
        fit_wrap(func, len(vecs), fit_parameters),
        vecs,
        dependent_variable,
        maxfev=20000,
        p0=guess,
        **kw
    )
