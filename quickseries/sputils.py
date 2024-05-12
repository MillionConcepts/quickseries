from typing import Any, Callable, Sequence, Union

import numpy as np
import sympy as sp

LmSig = Callable[[Any], Union[np.ndarray, float]]


def lambdify(
    func: Union[str, sp.Expr],
    modules: Union[str, Sequence[str]] = ("scipy", "numpy")
) -> LmSig:
    """
    Transform a sympy Expr or a string representation of a function into a
    callable with enforced argument order, incorporating code from specified
    modules.
    """
    if isinstance(func, str):
        try:
            func = sp.sympify(func)
        except sp.SympifyError:
            raise ValueError(f"Unable to parse {func}.")
    # noinspection PyTypeChecker
    return sp.lambdify(
        sorted(func.free_symbols, key=lambda x: str(x)), func, modules
    )
