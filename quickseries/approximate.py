from inspect import getfullargspec, signature
from itertools import chain
import re
from types import NoneType
from typing import Literal, Optional, Sequence, Union, Collection

from cytoolz import groupby
from dustgoggles.func import gmap
import numpy as np
import sympy as sp

from quickseries.bases import Orthobasis, BASIS_REGISTRY
from quickseries.expansions import (
    multivariate_taylor,
    series_lambda,
    orthoproject_1d,
    orthoproject_nd,
)
from quickseries.simplefit import fit
from quickseries.sourceutils import (
    _cacheget,
    _cacheid,
    _finalize_quickseries,
    lastline,
)
from quickseries.sputils import LmSig, lambdify


EXP_PATTERN = re.compile(r"\w+ ?\*\* ?(\d+)")
"""what exponentials in sympy-lambdified functions look like"""

TERM_PATTERN = re.compile(r"([+* (-]+|^)([\d.]+)(e[+\-]?\d+)?.*?([+* )]|$)")
"""
What numerical of sympy-lambdified (possibly rewritten) functions look like
"""


def is_simple_poly(expr: sp.Expr) -> bool:
    gens = sp.poly_from_expr(expr)[1]["gens"]
    return all(isinstance(g, sp.Symbol) for g in gens)


def regexponents(text: str) -> tuple[int]:
    # noinspection PyTypeChecker
    return tuple(map(int, re.findall(EXP_PATTERN, text)))


def _decompose(
    remaining: tuple[str],
    reduced: set[str],
    replacements: list[tuple[int, list[int]]],
) -> bool:
    if len(remaining) == 1:  # trivial case
        replacements[0][1][:] = [1 for _ in range(replacements[0][0])]
        return True
    counts = {
        k: len(v)
        for k, v in groupby(lambda x: x, remaining).items()
        if k not in reduced
    }
    if len(counts) < 2:  # nothing useful left to do
        return True
    elif counts[max(counts)] > 1:
        # don't decompose the biggest factor; because it appears more than
        # once, we'd like to precompute it
        reduced.add(max(counts))
        return False
    # otherwise, do a decomposition pass with the smallest factor
    factor = sorted(counts.keys())[0]
    for k, v in replacements:
        factorization = []
        # "divide out" `factor` from elements of existing decomposition
        for f in v:
            # don't decompose factors we've already evaluated, and don't
            # try to divide `factor` out of smaller factors (nonsensical)
            if f in reduced or f <= factor:
                factorization.append(f)
                continue
            factorization.append(factor)
            difference = f - factor
            while difference >= max([e for e in remaining if e != f]):
                factorization.append(factor)
                difference = difference - factor
            if difference > 0:
                factorization.append(difference)
        v[:] = factorization
    reduced.add(factor)
    return False


def optimize_exponents(
    exps: Sequence[int],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Given a sequence of integer exponents greater than 1 that appear in an
    expression, select exponents it might be useful to precompute in order to
    calculate the expression more quickly: any exponents that are 'repeated',
    even if only as an additive component of another exponent. More generally,
    this provides a minimal set of joint additive decompositions of the
    elements of `exps`.

    For instance, given 'exps=(2, 8, 9)' (meaning that the expression
    contains x ** 2, x ** 8, and x ** 9), optimize_exponents() will suggest:

    * precompute x2 = x * x
    * precompute x6 = x2 * x2 * x2 (note that this is only an intermediate
        variable)
    * replace x ** 2 with x2
    * replace x ** 8 with x2 * x6
    * replace x ** 9 with x * x2 * x6

    Note:
        This is typically called from`_rewrite_precomputed()`, meaning that
        `exps` are typically exponents that appear in the Horner form of a
        polynomial. Because Hornerization removes all repeated exponents that
        do not require precomputation to remove, in practice, there will often
        be no repeated exponents to precompute by this step, even if there
        are repeated exponents in the non-Horner form of the polynomial.

    Args:
        exps: Sequence of integers representing exponents of terms in a
            univariate expression.

    Returns:
        replacements: dict[int, list[int]] -- dict whose keys are elements of
            `exps` that should be precomputed, and whose values are lists
            containing keys of `variables` representing the exponents that
            should be used to precompute them. In the (2, 8, 9) example above,
            `replacements` would be `{2: [2], 8: [2, 6], 9: [2, 6, 1]}`.
        variables: dict[int, list[int]] -- dict whose keys are exponents that
            should be precomputed and substituted into the an expression, and
            whose values are lists whose elements are either `1` or other keys
            of `variables`, meaning the terms that should be used to
            precompute the corresponding exponent. In the (2, 8, 9) example
            above, `variables` would be `{2: [1, 1], 6: [2, 2, 2]}`.
    """
    # list of tuples like: (power, [powers to use in decomposition])
    replacements = [(e, [e]) for e in sorted(exps)]
    # which powers have we already assessed?
    reduced = set()
    # which powers haven't we?
    remaining = tuple(chain(*[r[1] for r in replacements]))
    # NOTE: _decompose() modifies 'remaining' and 'reduced' inplace
    # noinspection PyTypeChecker
    while _decompose(remaining, reduced, replacements) is False:
        remaining = tuple(chain(*[r[1] for r in replacements]))
    # this is analogous to casting to set: we no longer care about number of
    # occurrences
    replacements = {k: v for k, v in replacements}
    # figure out which factors we'd like to predefine as variables, and what
    # the "building blocks" of those variables are. 1 is a placeholder: we
    # will never define it, but it's useful in this loop.
    variables = {1: [1]}
    for e in sorted(set(remaining)):
        if e == 1:
            continue
        if exps.count(e) == 1:
            if not any(k > e for k, v in replacements.items()):
                continue
        vfactor, remainder = [], e
        while remainder > 0:
            pick = max([v for v in variables.keys() if v <= remainder])
            vfactor.append(pick)
            remainder -= pick
        variables[e] = vfactor
    # remove the placeholder first-order variable
    variables.pop(1)
    return replacements, variables


def force_line_precision(line: str, precision: Literal[16, 32, 64]) -> str:
    """
    Some library functions/classes, given a Python expression that contains
    Python `int` or `float` literals, will upcast arguments to some default
    precision (typically 64-bit). This function attempts to preempt that
    behavior.

    Given a Python expression as a string of source code,
    `force_line_precision()` wraps all numeric literals in that expression
    in numpy floating-point dtype constructors of bit width 'precision'.
    In most cases, this will prevent a function generated from the source
    code from upcasting its arguments past `precision`.

    For instance, `force_line_precision('x*(x*x*x*x*x + 5)', 32)`
    should return `'x*(x*x*x*x*x + float32(5.0))'`.

    Args:
        line: a line of Python source code, represented as a string
        precision: the floating-point bit width to enforce in `line`. Can be
            16, 32, or 64.

    Returns:
        A rewritten line of Python source code.
    """
    constructor_rep = f"float{precision}"
    constructor = getattr(np, f"float{precision}")
    last, out = 0, ""
    for match in re.finditer(TERM_PATTERN, line):
        out += line[last : match.span()[0]]
        # don't replace exponents
        if match.group(1) == "**":
            out += line[slice(*match.span())]
        else:
            # NOTE: casting number to string within the f-string statement
            # appears to upcast it before generating the representation.
            number = str(constructor(float(match.group(2))))
            out += f"{match.group(1)}{constructor_rep}({number}"
            if match.group(3) is not None:  # scientific notation
                out += match.group(3)
            out += f"){match.group(4)}"
        last = match.span()[1]
    return out + line[last:]


def _rewrite_precomputed(
    polyexpr: str, free: Collection[str]
) -> tuple[str, list[str]]:
    """
    Rewrite a polynomial with precomputed exponents. `optimize_exponents()`
    implements most of the domain logic; see that function's docstring for
    general rationale. This function performs the string manipulation and
    free variable handling that makes `optimize_exponents()` useful in the
    context of the `quickseries()` pipeline.

    This function is intended as a subroutine of `_rewrite()` and should
    generally only be called by that function.
    """
    # replacements: what factors we will decompose each exponent into
    # free: which factors we will define as variables, and their
    # "building blocks"
    factorlines = []
    for f in free:
        expat = re.compile(rf"{f}+ ?\*\* ?(\d+)")
        replacements, variables = optimize_exponents(
            gmap(int, expat.findall(polyexpr))
        )
        for k, v in variables.items():
            multiplicands = []
            for power in v:
                if power == 1:
                    multiplicands.append(f)
                else:
                    multiplicands.append(f"{f}{power}")
            factorlines.append(f"{f}{k} = {'*'.join(multiplicands)}")
        for k, v in replacements.items():
            substitution = "*".join([f"{f}{r}" if r != 1 else f for r in v])
            polyexpr = polyexpr.replace(f"{f}**{k}", substitution)
    return polyexpr, factorlines


def _rewrite(
    poly_lambda: LmSig,
    precompute: bool = True,
    precision: Optional[Literal[16, 32, 64]] = None,
) -> str:
    """
    Internal handler function for `quickseries.quickseries`. Given a
    lambdified-by-sympy function that computes a polynomial, extract its
    underlying polynomial expression, Hornerize it, and optionally /
    heuristically apply exponent precomputation and precision enforcement.

    This function should generally only be called by `_make_quickseries()`,
    which itself should generally only be called by `quickseries()`.
    """
    # sympy will always place this on a single line. it includes
    # the Python expression form of the hornerized polynomial
    # and a return statement. lastline() grabs polynomial and strips return.
    polyexpr = lastline(poly_lambda)
    # remove pointless '1.0' terms
    polyexpr = re.sub(r"(?:\*+)?1\.0\*+", "", polyexpr)
    # names of arguments to the lambdified function
    free = getfullargspec(poly_lambda).args
    lines = []
    if precompute is True:
        polyexpr, factorlines = _rewrite_precomputed(polyexpr, free)
        lines += factorlines
    if precision is not None:
        polyexpr = force_line_precision(polyexpr, precision)
    lines.append(f"return {polyexpr}")
    _, key = _cacheid()
    lines.insert(0, f"def {key}({', '.join(free)}):")
    return "\n    ".join(lines)


def _pvec(
    bounds: Sequence[tuple[float, float]], offset_resolution: int
) -> list[np.ndarray]:
    axes = [np.linspace(*b, offset_resolution) for b in bounds]
    indices = map(np.ravel, np.indices([offset_resolution for _ in bounds]))
    return [j[i] for j, i in zip(axes, indices)]


def _makebounds(
    bounds: Optional[Sequence[tuple[float, float]] | tuple[float, float]],
    n_free: int,
    point: Optional[Sequence[float] | float],
) -> tuple[list[tuple[float, float]], list[float]]:
    bounds = (-1, 1) if bounds is None else bounds
    if not isinstance(bounds[0], (list, tuple)):
        bounds = [bounds for _ in range(n_free)]
    if point is None:
        point = [np.mean(b) for b in bounds]
    elif not isinstance(point, (list, tuple)):
        point = [point for _ in bounds]
    return bounds, point


def _dispatch_series_expansion(
    basis: Orthobasis | None,
    bounds,
    expr,
    is_poly,
    nterms,
    point,
    add_coeffs,
    usequad,
    prunetol,
):
    order = len(bounds)
    if basis is None:
        if (order == 1) and (is_poly is False):
            approx, expr, nprune = series_lambda(
                expr, point[0], nterms, add_coeffs, prunetol=prunetol
            )
        else:
            approx, expr, nprune = multivariate_taylor(
                expr, point, nterms, add_coeffs, prunetol=prunetol
            )
    elif order == 1:
        approx, expr, nprune = orthoproject_1d(
            expr, nterms, basis, bounds[0], usequad, None, add_coeffs, prunetol
        )
    else:
        approx, expr, nprune = orthoproject_nd(
            expr,
            nterms,
            tuple(basis for _ in range(order)),
            bounds,
            usequad,
            add_coeffs,
            prunetol,
        )
    return approx, expr, nprune


def _perform_series_fit(
    expr: str | sp.Expr,
    bounds: Sequence[tuple[float, float]],
    nterms: int,
    fitres: int,
    point: float | Sequence[float],
    bound_series_fit: bool,
    is_poly: bool,
    basis: Orthobasis | None,
    usequad: bool,
    prunetol: float | None
) -> tuple[sp.Expr, np.ndarray, int]:
    lamb, vecs = lambdify(expr), _pvec(bounds, fitres)
    approx, expr, nprune = _dispatch_series_expansion(
        basis, bounds, expr, is_poly, nterms, point, True, usequad, prunetol
    )
    try:
        dep = lamb(*vecs)
    except TypeError as err:
        # this is a potentially slow but unavoidable case
        if "converted to Python scalars" not in str(err):
            raise
        dep = np.array([lamb(v) for v in vecs])
    guess = [1 for _ in range(len(signature(approx).parameters) - len(vecs))]
    bbox = (-1, 1) if basis == "legendre" else (-5, 5)
    params, _ = fit(
        func=approx,
        vecs=vecs,
        dependent_variable=dep,
        guess=guess,
        bounds=bbox if bound_series_fit is True else None,
    )
    # insert coefficients into polynomial
    expr = expr.subs({f"a_{i}": coef for i, coef in enumerate(params)})
    return expr, params, nprune


def _make_series(
    expr: sp.Expr,
    is_poly: bool,
    bounds: Sequence[tuple[float, float]],
    nterms: int,
    point: float | Sequence[float],
    basis: Orthobasis | None,
    usequad: bool,
    prunetol: float | None
):
    res = _dispatch_series_expansion(
        basis, bounds, expr, is_poly, nterms, point, False, usequad, prunetol
    )
    return res[1], res[2]


def apply_assumptions(expr, bounds):
    """
    substitute with variables containing quickseries' assumptions:
    they are (a) real, (b) if associated bounds are strictly > 0, positive,
    and if strictly < 0, negative
    """
    for s, b in zip(expr.free_symbols, bounds):
        assumptions = {"real": True}
        if min(b) > 0:
            assumptions["positive"] = True
        elif max(b) < 0:
            assumptions["negative"] = True
        sym = sp.Symbol(str(s), **assumptions)
        expr = expr.subs(s, sym)
    return expr


def _make_quickseries(
    approx_poly: bool,
    bound_series_fit: bool,
    bounds: Optional[Sequence[tuple[float, float]] | tuple[float, float]],
    expr: sp.Expr,
    fit_series_expansion: bool | None,
    fitres: int,
    nterms: int,
    point: Optional[Sequence[float] | float],
    precision: Optional[Literal[16, 32, 64]],
    prefactor: bool,
    basis: Orthobasis | None,
    usequad: bool,
    prunetol: float | None
) -> dict[str, sp.Expr | np.ndarray | str]:
    # TODO: auto-basis-selection logic w/taylor fallback

    if len(expr.free_symbols) == 0:
        raise ValueError("Function must have at least one free variable.")
    ndim = len(expr.free_symbols)
    if basis is not None and fit_series_expansion is None and ndim > 1:
        # numerical optimization pass on orthonormal basis projections of
        # expressions with > 1 free variable rarely produces valid results
        # (parameters are not meaningfully covariant) so is usually just a
        # waste of time or a way to introduce small amounts of numerical error
        fit_series_expansion = False
    elif fit_series_expansion is None:
        fit_series_expansion = True
    bounds, point = _makebounds(bounds, ndim, point)
    expr = apply_assumptions(expr, bounds)
    free = sorted(expr.free_symbols, key=lambda s: str(s))
    output, is_poly = {}, is_simple_poly(expr)
    output["orig_expr"] = expr
    kwargs = {
        "expr": expr,
        "bounds": bounds,
        "nterms": nterms,
        "is_poly": is_poly,
        "basis": basis,
        "point": point,
        "usequad": usequad,
        "prunetol": prunetol
    }
    if (approx_poly is True) or (is_poly is False):
        if fit_series_expansion is True:
            expr, output["params"], output["nprune"] = _perform_series_fit(
                **kwargs,
                fitres=fitres,
                bound_series_fit=bound_series_fit,
            )
        else:
            expr, output["nprune"] = _make_series(**kwargs)
    # rewrite polynomial in horner form for fast evaluation
    output["expr"] = sp.horner(expr)
    polyfunc = sp.lambdify(free, output["expr"], ("scipy", "numpy"))
    # polish it and optionally rewrite it to precompute repeated powers or
    # force precision
    return output | {"source": _rewrite(polyfunc, prefactor, precision)}


def quickseries(
    func: Union[str, sp.Expr],
    *,
    bounds: tuple[float, float] = (-1, 1),
    nterms: int = 9,
    point: Optional[float] = None,
    fitres: int = 100,
    prefactor: Optional[bool] = None,
    approx_poly: bool = False,
    jit: bool = False,
    precision: Optional[Literal[16, 32, 64]] = None,
    fit_series_expansion: bool | None = None,
    bound_series_fit: bool = False,
    extended_output: bool = False,
    cache: bool = True,
    basis: str | Orthobasis | None = "chebyshev",
    usequad: bool = True,
    prunetol: float | None = 1e-12
) -> Union[LmSig, tuple[LmSig, dict]]:
    basis = None if basis == "taylor" else basis
    if not isinstance(func, (str, sp.Expr)):
        raise TypeError(
            f"Unsupported type for func: expected str or Expr, "
            f"got {type(func)}."
        )
    if (
        not isinstance(basis, (Orthobasis, NoneType))
        and basis not in BASIS_REGISTRY
    ):
        raise ValueError(f"Unknown basis {basis}.")
    if isinstance(basis, str):
        basis = BASIS_REGISTRY[basis]
    polyfunc, ext = None, {"cache": "off"}
    if cache is True:
        polyfunc, source = _cacheget(jit)
        if polyfunc is not None:
            ext |= {"source": source, "cache": "hit"}
        else:
            ext["cache"] = "miss"
    if polyfunc is None:
        ext |= _make_quickseries(
            approx_poly,
            bound_series_fit,
            bounds,
            func if isinstance(func, sp.Expr) else sp.sympify(func),
            fit_series_expansion,
            fitres,
            nterms,
            point,
            precision,
            prefactor if prefactor is not None else not jit,
            basis,
            usequad,
            prunetol
        )
        polyfunc = _finalize_quickseries(ext["source"], jit, cache)
    if extended_output is True:
        return polyfunc, ext
    return polyfunc
