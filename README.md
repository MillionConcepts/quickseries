# quickseries

`quickseries` generates Python functions that perform fast vectorized power 
series approximations of locally-continuous univariate mathematical functions. 
`quickseries`is in alpha; bug reports are appreciated.

Install from source using `python setup.py`. Dependencies are also described
in a Conda `environment.yml` file.

Further documentation forthcoming.

## example of use

```
>>> from timeit import timeit
>>> import numpy as np
>>> from quickseries import quickseries

>>> approx = quickseries(
...    "sin(x)*cos(x)", x0=0, n_terms=12, bounds=[-np.pi, np.pi]
... )
>>> x = np.linspace(-3.14, 3.14, 100000)
>>> print(f"max error: {max(abs(np.sin(x) * np.cos(x) - approx(x)))}")
>>> print("original runtime:")
>>> %timeit np.sin(x) * np.cos(x)
>>> print("approx runtime:")
>>> %timeit approx(x)

max error: 0.00032709055456955904
original runtime:
965 µs ± 2.29 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
approx runtime:
318 µs ± 4.36 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```



