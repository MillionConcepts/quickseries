import pickle
from hashlib import md5
from inspect import currentframe, getargvalues, getsource
import linecache
from pathlib import Path
import re
from types import FunctionType
from typing import Callable

from dustgoggles.dynamic import define, get_codechild


CACHE_ARGS = (
    "func",
    "bounds",
    "nterms",
    "point",
    "fitres",
    "prefactor",
    "approx_poly",
    "precision",
    "fit_series_expansion",
    "bound_series_fit"
)


def cache_source(source: str, fn: Path):
    fn = str(fn)
    linecache.cache[fn] = (len(source), None, source.splitlines(True), fn)


# TODO: pull this little bitty change up to dustgoggles
def compile_source(source: str, fn: str = ""):
    return get_codechild(compile(source, fn, "exec"))


def _cachedir(callfile: str) -> Path:
    if callfile == 'ipython_shell':
        import IPython.paths

        return Path(IPython.paths.get_ipython_cache_dir()) / "qs_cache"
    return Path(callfile).parent / "__pycache__" / "qs_cache"


def _cachekey(args, callfile=None):
    from quickseries import __version__

    # TODO: is this actually stable?
    arghash = pickle.dumps(
        {a: args.locals[a] for a in sorted(CACHE_ARGS)}
        | {'f': callfile, '__version__': __version__}
    )
    # arbitrary cutoff for a reasonable tradeoff between collision safety and
    # readability
    return f"quickseries_{md5(arghash).hexdigest()}"[:-18]


# TODO, maybe: the frame traversal is potentially wasteful when repeated,
#  although it probably doesn't matter too much.
def _cacheid():
    """
    WARNING: do not call this outside the normal quickseries workflow. It can
     be tricked, but to no good end.
    """
    frame, callfile, args = currentframe(), None, None
    while callfile is None:
        frame = frame.f_back
        if frame is None or frame.f_code.co_filename == "<stdin>":
            callfile = "__quickseries_anonymous_caller_cache__/anonymous"
        elif hasattr(frame.f_code, "co_name"):
            if args is None and frame.f_code.co_name == "quickseries":
                args = getargvalues(frame)
            elif frame.f_code.co_name == "benchmark":
                continue
            elif args is not None:
                callfile = frame.f_code.co_filename
                if re.search(r"interactiveshell.py|ipython", callfile):
                    callfile = 'ipython_shell'
    if args is None:
        raise ReferenceError("Cannot use _cachefile() outside quickseries().")
    key = _cachekey(args, callfile)
    return _cachedir(callfile) / key / "func", key


def _compile_quickseries(source, jit, cache, cachefile):
    globals_ = globals()
    if (precmatch := re.search(r"float\d\d", source)) is not None:
        import numpy

        globals_[precmatch.group()] = getattr(numpy, precmatch.group())
    func = FunctionType(compile_source(source, str(cachefile)), globals_)
    cache_source(source, cachefile)
    func.__doc__ = source
    if jit is True:
        import numba as nb

        return nb.njit(func, cache=cache)
    return func


def _cacheget(jit=False):
    cachefile, key = _cacheid()
    if not cachefile.exists():
        return None, None
    with cachefile.open() as stream:
        source = stream.read()
    return _compile_quickseries(source, jit, True, cachefile), source


def _cachewrite(source, cachefile):
    # we make the __pycache__ directory to enable numba JIT result caching,
    # just in case it happens; if it doesn't, the presence of the directory is
    # harmless.
    (cachefile.parent / "__pycache__").mkdir(exist_ok=True, parents=True)
    # TODO, maybe: use a more sensible data structure
    with cachefile.open("w") as stream:
        stream.write(source)


def _finalize_quickseries(source, jit=False, cache=False):
    # note that we use this as a function identifier and 'fake' target for
    # linecache even if we're not actually using the quickseries cache
    cachefile, key = _cacheid()
    if cache is True:
        _cachewrite(source, cachefile)
    return _compile_quickseries(source, jit, cache, cachefile)


def lastline(func: Callable) -> str:
    """try to get the last line of a function, sans return statement"""
    return tuple(
        filter(None, getsource(func).split("\n"))
    )[-1].replace("return", "").strip()
