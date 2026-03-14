from functools import reduce
from itertools import product
from typing import Union, Sequence

import numpy as np
from dustgoggles.structures import listify
import sympy as sp

from quickseries.bases import Orthobasis
from quickseries.sputils import LmSig


def _rectify_series(
    series, add_coefficients: bool, prunetol: float | None = None
):
    # if isinstance(series, int):
    #     series = sp.sympify(series)
    if isinstance(series, sp.Order):
        raise ValueError(
            "Cannot produce a meaningful approximation with the requested "
            "parameters (most likely order is too low)."
        )
    outargs, coefsyms = [], []

    def maybe_prune(thing):
        try:
            if prunetol is None or abs(thing) > prunetol:
                return False
            return True
        except (ValueError, TypeError):
            return False

    nprune = 0
    for a in series.args:
        # NOTE: the Expr.evalf() calls are simply to try to evaluate
        #  anything we can.
        if hasattr(a, "evalf") and isinstance((n := a.evalf()), sp.Number):
            # constant term case
            if maybe_prune(n):
                nprune += 1
                continue
            outargs.append(n)
        elif isinstance(a, sp.Order):
            # do not care, we know it's a series
            continue
        elif isinstance(a, (sp.Mul, sp.Symbol, sp.Pow)):
            if maybe_prune(n := a.evalf()):
                nprune += 1
                continue
            if add_coefficients is True:
                coefficient = sp.symbols(f"a_{len(coefsyms)}")
                outargs.append(coefficient * n)
                coefsyms.append(coefficient)
            else:
                outargs.append(n)
        else:
            raise ValueError(
                f"don't know how to handle expression element {a} of "
                f"type({type(a)})"
            )
    return sum(outargs), coefsyms, nprune


def series_lambda(
    func: Union[str, sp.Expr],
    x0: float = 0,
    nterms: int = 9,
    add_coefficients: bool = False,
    modules: Union[str, Sequence[str]] = ("scipy", "numpy"),
    prunetol: float | None = 1e-12
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
    expr, coefsyms, nprune = _rectify_series(
        series, add_coefficients, prunetol
    )
    syms = sorted(func.free_symbols, key=lambda x: str(x))
    # noinspection PyTypeChecker
    return sp.lambdify(syms + coefsyms, expr, modules), expr, nprune


def additive_combinations(n_terms, number):
    if n_terms == 1:
        return [(n,) for n in range(number + 1)]
    combinations = []  # NOTE: this is super gross-looking written as a chain
    for j in range(number + 1):
        combinations += [
            (j, *t) for t in additive_combinations(n_terms - 1, number - j)
        ]
    return combinations


def multivariate_taylor(
    func: Union[str, sp.Expr],
    point: Sequence[float],
    nterms: int,
    add_coefficients: bool = False,
    prunetol: float | None = 1e-12
) -> tuple[LmSig, sp.Expr]:
    func = sp.sympify(func) if isinstance(func, str) else func
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
        sp.Mul, [(x - a) ** i for x, a, i in zip(argsyms, pointsyms, ixsyms)]
    )
    taylor = deriv / fact * err
    # TODO, probably: there's a considerably faster way to do this in some
    #  cases by precomputing partial derivatives
    decomp = additive_combinations(dimensionality, nterms - 1)
    built = reduce(
        sp.Add,
        (taylor.subs({i: d for i, d in zip(ixsyms, d)}) for d in decomp),
    ).doit()
    evaluated = built.subs({s: p for s, p in zip(pointsyms, point)}).evalf()
    # this next line is kind of aesthetic -- we just want the argument names
    # to be consistent with the input
    evaluated = evaluated.subs({a: p for a, p in zip(argsyms, pointsyms)})
    evaluated, coefsyms, nprune = _rectify_series(
        evaluated, add_coefficients, prunetol
    )
    # noinspection PyTypeChecker
    return sp.lambdify(pointsyms + coefsyms, evaluated), evaluated, nprune


def orthoproject_1d(
    func: sp.Expr,
    nterms: int,
    basis: Orthobasis,
    bounds: tuple[float, float],
    use_quadrature=False,
    quad_order=None,
    add_coefficients=False,
    prunetol: float | None = 1e-12
):
    xi: sp.Symbol = next(iter(func.free_symbols))
    f_on_std = basis.transform_input(func, *bounds, xi)
    # noinspection PyTypeChecker
    coeffs = []
    polys = [basis.orthopoly(n, xi) for n in range(nterms)]

    if use_quadrature:
        # default oversampling
        quad_order = quad_order or nterms * 2
        pts, wts = basis.quadrature(quad_order)

        for n, Pn in enumerate(polys):
            values = [
                w * f_on_std.subs(xi, pt) * Pn.subs(xi, pt)
                for pt, w in zip(pts, wts)
            ]
            coeff = sum(values) / basis.norm(n)
            coeffs.append(coeff)
    else:
        for n, Pn in enumerate(polys):
            integrand = f_on_std * Pn * basis.weight_fn(xi)
            coeff = sp.integrate(integrand, (xi, *basis.domain)) / basis.norm(n)
            coeffs.append(coeff.evalf())
    if prunetol is not None:
        # we do our own pruning rather than leaving it to _rectify_series() --
        # there are never any of the really gross cases here, so it's faster
        # and cleaner to just do it during assembly
        comps = [c * Pn for c, Pn in zip(coeffs, polys) if abs(c) > prunetol]
        poly_std = sum(comps)
        nprune = len(coeffs) - len(comps)
    else:
        poly_std = sum(c * Pn for c, Pn in zip(coeffs, polys))
        nprune = 0
    # noinspection PyTypeChecker
    poly_on_orig = basis.transform_output(poly_std, *bounds, xi)
    expr, coefsyms, _ = _rectify_series(poly_on_orig, add_coefficients)
    x = tuple(func.free_symbols)[0]
    return sp.lambdify([x, *coefsyms], expr), expr, nprune


def orthoproject_nd(
    func: sp.Expr,
    nterms: int,
    bases: tuple[Orthobasis, ...],
    bounds: tuple[tuple[float, float]],
    usequad=True,
    add_coefficients=False,
    prunetol = 1e-12
):
    if not usequad:
        raise NotImplementedError("Fully analytic N-D not implemented")

    syms = sorted(func.free_symbols, key=lambda s: str(s))
    assert len(syms) == len(bases) == len(bounds)

    # Transform function to standard domain
    f_std = func
    for s, b, (a, c) in zip(syms, bases, bounds):
        f_std = b.transform_input(f_std, a, c, s)

    polys = [
        [b.orthopoly(n, s) for n in range(nterms)]
        for b, s in zip(bases, syms)
    ]
    quads = [b.quadrature(nterms * 2) for b in bases]
    all_points = [q[0] for q in quads]
    all_weights = [q[1] for q in quads]

    coeffs = []
    degs_series = [
        degs for degs in product(range(nterms), repeat=len(syms))
        if sum(degs) <= nterms
    ]

    for degs in degs_series:
        comps = [polys[i][d] for i, d in enumerate(degs)]
        coeff = 0
        for p_comb, w_comb in zip(product(*all_points), product(*all_weights)):
            subs_dict = {s: p for s, p in zip(syms, p_comb)}
            f_val = f_std.subs(subs_dict)
            term_val = f_val * np.prod(
                [comp.subs(syms[i], p_comb[i]) for i, comp in enumerate(comps)]
            )
            coeff += np.prod(w_comb) * term_val
        coeff /= np.prod([b.norm(d) for b, d in zip(bases, degs)])
        coeffs.append(coeff.evalf())

    final_terms = []
    # we do our own pruning rather than leaving it to _rectify_series() --
    # there are never any of the really gross cases here, so it's faster
    # and cleaner to just do it during assembly
    for degs, coeff in zip(degs_series, coeffs):
        if prunetol is not None and abs(coeff) < prunetol:
            continue
        poly_term = coeff * sp.Mul(*[polys[i][d] for i, d in enumerate(degs)])
        final_terms.append(poly_term)
    final_poly_std = sp.Add(*final_terms)
    final_poly_orig = final_poly_std
    for s, b, (a, c) in zip(syms, bases, bounds):
        final_poly_orig = b.transform_output(final_poly_orig, a, c, s)

    final_poly, coefsyms, _ = _rectify_series(
        final_poly_orig, add_coefficients
    )
    return (
        sp.lambdify(syms + coefsyms, final_poly_orig),
        final_poly_orig,
        len(degs_series) - len(final_terms)
    )
