# quickseries

`quickseries` generates Python functions that perform fast vectorized power 
series approximations of mathematical functions. It can provide performance 
improvements ranging from ~3x (simple functions, no fiddling around with 
parameters) to ~100x (complicated functions, some parameter tuning).

`quickseries` is in beta; bug reports are appreciated.

Install from source using `pip install .`. Dependencies are also described
in a Conda `environment.yml` file.

Examples and tips follow. Further documentation forthcoming.

## example of use

```
>>> import numpy as np
>>> from quickseries import quickseries

>>> bounds = (-np.pi, np.pi)
>>> approx = quickseries("sin(x)*cos(x)", point=0, order=12, bounds=bounds)
>>> x = np.linspace(*bounds, 100000)
>>> print(f"max error: {max(abs(np.sin(x) * np.cos(x) - approx(x)))}")
>>> print("original runtime:")
>>> %timeit np.sin(x) * np.cos(x)
>>> print("approx runtime:")
>>> %timeit approx(x)

max error: 0.0003270875375037813
original runtime:
968 µs ± 2.17 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
approx runtime:
325 µs ± 3.89 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

## limitations

* `quickseries` only works for functions ℝ<sup>_n_</sup>🡒ℝ for finite _n_. In
  programming terms, this means it will only produce functions that accept a 
  fixed number of floating-point or integer arguments (which may be 'arraylike'
  objects such as pandas `Series` or numpy `ndarrays`) and return a single 
  floating-point value (or a 1-D floating-point array if passed arraylike 
  arguments).
* `quickseries` only works consistently on functions that are continuous and 
  infinitely differentiable within the domain of interest. Specifically, they
  should not have singularities, discontinuities, or infinite / undefined 
  values at `point` or within `bounds`. Failure cases differ:
  * `quickseries` will always fail on functions that are infinite/undefined 
    at `point`, like `quickseries("ln(x)", point=-1)`.
  * It will almost always fail on functions with a largeish interval of 
    infinite/undefined values within `bounds`, such as
    `quickseries("gamma(x)", bounds=(-1.1, 0), point=-0.5)`.
  * It will usually succeed but produce bad results on functions with 
    singularities or point discontinuities within `bounds` or 
    near `point` but not at `point`, such as `quickseries("tan(x)", bounds=(1, 2))`.
  * It will often succeed, but usually produce bad results, on univariate 
    functions that are continuous but not differentiable at `point`, such as 
    `quickseries("abs(sin(x))", point=0)`. It will always fail on multivariate 
    functions of this kind.
* Functions given to `quickseries` must be expressed in strict closed form 
  and include only finite terms. They cannot contain limits, integrals, 
  derivatives, summations, continued fractions, etc.  
* `quickseries` is not guaranteed to work for all such functions.

## tips

* Multivariate `quickseries()`-generated functions always map positional arguments
  to variables in the string representation of the input function in alphanumeric
  order. This is in order to maintain consistency between slightly different 
  forms of the same expression.
  * Examples:
    * `quickseries("cos(x) * sin(y)")(1, 2)` approximates `sin(1) * cos(2)`
    * `quickseries("sin(y) * cos(x)")(1, 2)` approximates `cos(1) * sin(2)` 
    * `quickseries("sin(x) * cos(y)")(1, 2)` approximates `sin(1) * cos(2)`
  * Note that you can always determine the argument order of a `quickseries()`-
    generated function by using the `help()` builtin, `inspect.getfullargspec()`,
    examining the function's docstring, etc.
* Most legal Python variable names are allowable names for free variables.
  Named mathematical functions and constants are the major exceptions.
  * Examples:
    * `"ln(_)"`, `"ln(One_kitty)"`, `"ln(x0)"`, and `"ln(ă)"` will all work fine.
    * `"ln(if)"` and `"ln(🔥)"` will both fail, because `if` and `🔥` are not
      legal Python variable names.
    * `"ln(gamma)"` will fail, because `quickseries()` will interpret "gamma"
      as the gamma function.
    * `"cos(x) * cos(pi * 2)"` will succeed, but `quickseries()` will interpret 
      it as "the cosine of a variable named 'x' times the cosine of two times
      the mathematical constant pi" -- in other words, as `"cos(x)"`.
  * `quickseries.benchmark()` offers an easy way to test the accuracy and
  efficiency of `quickseries.quickseries()`-generated functions.
* Narrowing `bounds` will tend to make the approximation more accurate within
those bounds. In the example above, setting `bounds` to `(-1, 1)` provides 
~20x greater accuracy within the (-1, 1) interval (with the downside that 
the resulting approximation will get pretty bad past about +/-pi/2).
    * Like many optimizers, `quickseries()` tends to be much more effective 
      closer to 0 and when its input arguments have similar orders of 
      magnitude. If it is practical to shift/squeeze your data towards 0, you
      may be able to get more use out of `quickseries`. This is largely due to
      the fact that high-order polynomials are more numerically stable with 
      smaller input values.
    * Functions with a pole at 0, of course -- or whose series expansions have
      a pole at 0 -- can present an exception to this rule. It will still
      generally be better to keep their input values small.
* Increasing `nterms` will tend to make the approximation slower but more 
accurate. In the example above, increasing `nterms` to 14 provides ~20x 
greater accuracy but makes the approximation ~20% slower.
  * This tends to have diminishing returns. In the example above, increasing 
  `nterms` to 30 provides no meaningful increase in accuracy over `order=14`, 
  but makes the approximation *slower* than `np.sin(x) * np.cos(x)`.
  * Setting `nterms` too high can also cause the approximation algorithm to
  fail entirely.
  * The location of accuracy/performance "sweet spots" in the parameter space 
  depends on the function and the approximation bounds. If you want to 
  seriously optimize a particular function in a particular interval, you will 
  need to play around with these parameters.
* The speedup (or lack thereof) that a `quickseries()`-generated approximation 
  provides can vary greatly in different operating environments and on different 
  processors.
* It can also vary depending on the length of the input arguments. It generally 
  provides most benefit on arrays with tens or hundreds of thousands of elements,
  although this again varies depending on operating environment, the particular
  approximated function, etc.
* In general, `quickseries` provides more performance benefits for more 'complicated'
  input functions. This is due to the implicit 'simplification' offered by the 
  power series expansion.
* For most functions, placing `point` in the middle of `bounds` will produce the
best results, and if you don't pass `point` at all, `quickseries` defaults to 
placing it in the middle of `bounds`.
* It is often difficult to generate a polynomial approximation that
  remains good across a wide range of input values. In some cases, it may be 
  useful to generate different functions for different parts of your code, or 
  even to perform piecewise operations with multiple functions (although this 
  of course adds complexity and overhead).
* Functions generated by `quickseries()` may in some cases be less 
space/memory-efficient even if they are more time/compute-efficient.
* By default, if you pass a simple polynomial expression to `quickseries()`
(e.g. `"x**4 + 2 * x**3"`), it does not actually generate an approximation, 
but instead simply attempts to rewrite it in a more efficient form.
    * `nterms`, `bounds`, and `point` are ignored in this "rewrite" mode.
    * This type of `quickseries()`-generated function should produce the same 
    results as any other Python function that straightforwardly implements a
    form of the input polynomial (down to floating-point error).
    * This can produce surprising speedups even in simple cases -- for example,
    `quickseries("x**4")` is ~20x faster than `lambda x: x ** 4` on some 
    `numpy` arrays.  
    * If you want `quickseries()` to actually create an approximation of a 
    simple polynomial, pass `approx_poly=True`.
      * When approximating a polynomial, there is generally no good reason to 
      set `nterms` > that polynomial's order. If you do, the function 
      `quickseries()` generates will typically be very similar to a simple 
      rewrite of the input polynomial, but with slightly worse performance and 
      accuracy.
      * `point=0` often produces boring results for polynomial approximation.
* `quickseries()` is also capable of auto-jitting the functions it generates
with `numba`. Pass the `jit=True` argument. `numba` is an optional dependency; 
install it with your preferred package manager.
  * In many, but not all, cases, this will provide a significant performance
    improvement, sometimes by an order of magnitude. It also permits calling
    `quickseries`-generated functions from within other `numba`-compiled
    functions.
  * In addition to the other inconveniences that may arise from just-in-time
  compilation, some functions that work well without `numba` may not work well
  with `numba`.
* By default, `quickseries()` caches the code it generates. If you wish to 
  turn this behavior off, pass `cache=False`.
  * If you call `quickseries()` with the same arguments from separate modules, 
    it will write separate caches for each module.
  * ipython/Jupyter shells/kernels all share one cache within the same user 
    account.
  * `quickseries()` treats stdin or similar 'anonymous' invocation contexts 
    like modules named "__quickseries_anonymous_caller_cache__" in the current 
    working directory.
  * In this mode, `quickseries()` also caches the results of `numba` JIT 
    compilation, if it is active.
  * Caching is turned _off_ by default for `benchmark()`.
* If you pass the `precision` argument to `quickseries()`, it will attempt to
  guarantee that the function it returns will not cast input values to bit widths
  greater than the value of `precision`. Legal values of `precision` are 16, 32, 
  and 64. The returned function will not, however, attempt to reduce the precision
  of its arguments. For instance, `quickseries("sin(x) + exp(x)", precision=32)`
  will return a Python `float` if passed an `float`, and a `np.float64` `ndarray`
  if passed a `np.float64` `ndarray`. However, it will return a `np.float32`
  `ndarray` if passed a `np.float32` `ndarray`, which is not guaranteed without
  the `precision=32` argument. 
  * This can lead to significant speedups and memory usage improvements in
    cases where you do not need the extra precision.
  * Note that many libraries and formats do not support the "half-float" 
    values generated by `quickseries` when passed `precision=16`. 
* `quickseries` tends to be most effective on univariate functions, mostly 
   because the number of terms in a function's power expansion increases 
   geometrically with its number of free parameters.
* By default, `quickseries` takes the analytic series expansion of the input 
  function as a strong suggestion rather than the last word on the topic, and
  performs a numerical optimization step to improve its goodness of fit across
  `bounds`. There are good reasons you might not want it to do this, though --
   for instance, if your input arguments are always going to be quite close to 
  `point`, messing with the analytic series expansion may be wasteful or even
  counterproductive. If you don't want it to do this, pass `fit_series_expansion=False`.
  In this case, `quickseries` ignores the `bounds` argument, except to infer
  a value for `point` if you do not specify one.
  * In some cases, this optimization step can become numerically unstable. In
    these cases, you may wish to experiment with constraining it rather than 
    turning it off completely. You can do this by passing `bound_series_fit=True`.
* By default, the functions that `quickseries` generates precompute all repeated
  exponents in the generated polynomial. This is a space-for-time trade, and
  may not always be desirable (or even effective). You can turn this off by 
  passing `prefactor=False`. 
  * If `jit=True`, `quickseries` does _not_ do this by default. The `numba` 
    compiler implicitly performs a similar optimization, and computing these
    terms explicitly tends to be counterproductive. If you want `quickseries`
    to do it anyway, you can pass `prefactor=True`.