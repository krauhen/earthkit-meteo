# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Iterable, List, Union

import numpy as np
from earthkit.utils.array import array_namespace

_METHODS = ("sort", "numpy_bulk", "numpy", "hazen")


def _hazen_positions(n: int) -> List[np.float32]:
    """Cumulative plotting positions ``(i - 1/2) / n`` of ``n`` equally weighted members.

    Accumulated in single precision, member by member, with one rounding per member as in
    fieldextra (``fxtr_operator_probability.f90:1318-1323``): ``cum[0] = step`` with
    ``step = 0.5 * (1 / n)``, and ``cum[i] = cum[i-1] + 0.5 * (1 / n) * (w + w)`` with ``w = 1 / n``,
    which for equal weights is ``cum[i-1] + 2 * step``, an exact product. This is neither the
    closed form ``(i + 0.5) / n`` nor ``(cum + step) + step``: they differ in the last float32 bit
    and move some interpolated values by one packing step.

    The FMA that fieldextra's compiler may use in ``a + (b - a) * frac`` is deliberately not
    reproduced (known-value-mismatches #14).
    """
    f32 = np.float32
    step = f32(0.5) * (f32(1.0) / f32(n))
    cum = [step]
    for _ in range(1, n):
        cum.append(f32(cum[-1] + f32(step * f32(2.0))))
    return cum


def _hazen_quantile(arr_sorted, finite, cum, q, dim, xp, device):
    """One Hazen quantile ``q`` (a float32 fraction) of members sorted along ``dim``."""
    f32 = np.float32
    n = len(cum)
    i = next((k for k, c in enumerate(cum) if c > q), None)

    def member(k):
        return xp.take(arr_sorted, xp.asarray(k, device=device), axis=dim)

    if i is None:
        result = member(n - 1)
    elif i == 0:
        result = member(0)
    else:
        lo = member(i - 1)
        frac = f32(f32(q - cum[i - 1]) / f32(cum[i] - cum[i - 1]))
        result = lo + (member(i) - lo) * float(frac)
    return xp.where(finite, result, xp.nan)


def iter_quantiles(
    arr: np.ndarray,
    which: Union[int, List[float]] = 100,
    dim: int = 0,
    method: str = "sort",
) -> Iterable[np.ndarray]:
    """Iterate over the quantiles of a large array.

    Parameters
    ----------
    arr: numpy array
        Data array
    which: int or list of floats
        List of quantiles to compute, e.g. `[0., 0.25, 0.5, 0.75, 1.]`, or
        number of evenly-spaced intervals (e.g. 100 for percentiles).
    dim: int
        Dimension index along which to compute the quantiles.
    method: 'sort', 'numpy_bulk', 'numpy', 'hazen'
        Method of computing the quantiles:
        * sort: sort `arr` in place, then interpolates the quantiles one by one
        * numpy_bulk: compute all the quantiles at once using `numpy.quantile`
        * numpy: compute the quantiles one by one using `numpy.quantile`
        * hazen: the equal-weight Hazen rule of fieldextra (plotting positions ``(i - 1/2) / n``
          for the ``i``-th smallest of ``n`` members, linear interpolation between the two
          neighbouring members, the minimum below the first and the maximum above the last
          position). Computed in float32 whatever the input dtype, the result is float32,
          and it is NaN wherever any member is not finite. ``which`` holds fractions in [0, 1]
          (a quantile of 90 % is ``0.9``).

    Returns
    -------
    Iterable[numpy array]
        Quantiles, in increasing order if `which` is an `int`, otherwise in the order specified
    """
    if method not in _METHODS:
        raise ValueError(f"Invalid method {method!r}, expected 'sort', 'numpy_bulk', 'numpy', or 'hazen'")

    xp = array_namespace(arr)
    arr = xp.asarray(arr)
    if method == "hazen":
        arr = xp.asarray(arr, dtype=xp.float32)
    device = xp.device(arr)

    if isinstance(which, int):
        n = which
        qs = xp.linspace(0.0, 1.0, n + 1, device=device, dtype=arr.dtype)
    else:
        qs = xp.asarray(which, device=device, dtype=arr.dtype)

    if method == "numpy_bulk":
        quantiles = xp.quantile(arr, qs, axis=dim)
        yield from quantiles
        return

    if method == "hazen":
        finite = xp.all(xp.isfinite(arr), axis=dim)
        arr = xp.sort(arr, axis=dim)
        cum = _hazen_positions(arr.shape[dim])
        for q in qs:
            with np.errstate(invalid="ignore", over="ignore"):  # inf members are masked below
                yield _hazen_quantile(arr, finite, cum, np.float32(float(q)), dim, xp, device)
        return

    if method == "sort":
        arr = xp.asarray(arr)
        arr = xp.sort(arr, axis=dim)
        missing = xp.any(xp.isnan(arr), axis=dim)

    for q in qs:
        if method == "numpy":
            q = xp.asarray(q, dtype=arr.dtype)
            yield xp.quantile(arr, q, axis=dim)

        elif method == "sort":
            m = arr.shape[dim]
            f = (m - 1) * q
            j = int(f)
            x = f - j
            quantile = xp.take(arr, xp.asarray(j, device=device), axis=dim) * (1 - x)
            tmp = xp.take(arr, xp.asarray(min(j + 1, m - 1), device=device), axis=dim) * x
            quantile += tmp
            quantile = xp.where(missing, xp.nan, quantile)
            yield quantile
