from functools import reduce
from typing import Union, Sequence

from dustgoggles.structures import listify
import sympy as sp

from quickseries.sputils import LmSig


def _rectify_series(series, add_coefficients):
    if isinstance(series, sp.Order):
        raise ValueError(
            "Cannot produce a meaningful approximation with the requested "
            "parameters (most likely order is too low)."
        )
    outargs, coefsyms = [], []
    for a in series.args:
        # NOTE: the Expr.evalf() calls are simply to try to evaluate
        #  anything we can.
        if hasattr(a, "evalf") and isinstance((n := a.evalf()), sp.Number):
            outargs.append(n)
        elif isinstance(a, sp.Order):
            continue
        elif isinstance(a, (sp.Mul, sp.Symbol, sp.Pow)):
            if add_coefficients is True:
                coefficient = sp.symbols(f"a_{len(coefsyms)}")
                outargs.append((coefficient * a).evalf())
                coefsyms.append(coefficient)
            else:
                outargs.append(a.evalf())
        else:
            raise ValueError(
                f"don't know how to handle expression element {a} of "
                f"type({type(a)})"
            )
    return sum(outargs), coefsyms


def series_lambda(
    func: Union[str, sp.Expr],
    x0: float = 0,
    nterms: int = 9,
    add_coefficients: bool = False,
    modules: Union[str, Sequence[str]] = ("scipy", "numpy")
) -> tuple[LmSig, sp.Expr]:
    """
    Construct a power expansion of a sympy Expr or the string expression of a
    function; optionally, add free coefficients to the terms of the resulting
    polynomial to permit optimization by downstream functions.

    Args:
        func: Mathematical function to expand, expressed as a string or a
            sympy Expr.
        x0: Point about which to expand func.
        nterms: Order of power expansion.
        add_coefficients: If True, add additional arguments/symbols to the
            returned function and Expr corresponding to the polynomial's
            coefficients.
        modules: Modules from which to draw the building blocks of the
            returned function.

    Returns:
        approximant: Python function that implements the power expansion.
        expr: sympy Expr used to construct approximant.
    """
    func = sp.sympify(func) if isinstance(func, str) else func
    # limiting precision of x0 is necessary due to a bug in sp.series
    series = sp.series(func, x0=round(x0, 6), n=nterms)
    # noinspection PyTypeChecker
    # remove Order (limit behavior) terms, try to split constants from
    # polynomial terms
    expr, coefsyms = _rectify_series(series, add_coefficients)
    syms = sorted(func.free_symbols, key=lambda x: str(x))
    # noinspection PyTypeChecker
    return sp.lambdify(syms + coefsyms, expr, modules), expr


def additive_combinations(n_terms, number):
    if n_terms == 1:
        return [(n,) for n in range(number + 1)]
    combinations = []  # NOTE: this is super gross-looking written as a chain
    for j in range(number + 1):
        combinations += [
            (j, *t)
            for t in additive_combinations(n_terms - 1, number - j)
        ]
    return combinations


def multivariate_taylor(
    func: Union[str, sp.Expr],
    point: Sequence[float],
    nterms: int,
    add_coefficients: bool = False
) -> tuple[LmSig, sp.Expr]:
    if not isinstance(func, sp.Expr):
        func = sp.sympify(func)
    pointsyms = sorted(func.free_symbols, key=lambda s: str(s))
    dimensionality = len(pointsyms)
    argsyms = listify(
        sp.symbols(",".join([f"x{i}" for i in range(dimensionality)]))
    )
    ixsyms = listify(
        sp.symbols(",".join(f"i{i}" for i in range(dimensionality)))
    )
    deriv = sp.Derivative(func, *[(p, i) for p, i in zip(pointsyms, ixsyms)])
    # noinspection PyTypeChecker
    fact = reduce(sp.Mul, [sp.factorial(i) for i in ixsyms])
    err = reduce(
        sp.Mul,
        [(x - a) ** i for x, a, i in zip(argsyms, pointsyms, ixsyms)]
    )
    taylor = deriv / fact * err
    decomp = additive_combinations(dimensionality, nterms - 1)
    built = reduce(
        sp.Add,
        (taylor.subs({i: d for i, d in zip(ixsyms, d)}) for d in decomp)
    ).doit()
    evaluated = built.subs({s: p for s, p in zip(pointsyms, point)}).evalf()
    # this next line is kind of aesthetic -- we just want the argument names
    # to be consistent with the input
    evaluated = evaluated.subs({a: p for a, p in zip(argsyms, pointsyms)})
    evaluated, coefsyms = _rectify_series(evaluated, add_coefficients)
    # noinspection PyTypeChecker
    return sp.lambdify(pointsyms + coefsyms, evaluated), evaluated
