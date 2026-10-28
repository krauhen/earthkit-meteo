# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any, TypeAlias

from earthkit.utils.array import array_namespace

from earthkit.meteo import constants

ArrayLike: TypeAlias = Any

__all__ = ["moisture_flux_convergence"]


def _pad_nan(xp, a):
    """Surround a (nj-2, ni-2) array with one row/column of nan on each side."""
    col = xp.full((a.shape[0], 1), xp.nan, dtype=a.dtype, device=a.device)
    a = xp.concat([col, a, col], axis=1)
    row = xp.full((1, a.shape[1]), xp.nan, dtype=a.dtype, device=a.device)
    return xp.concat([row, a, row], axis=0)


def moisture_flux_convergence(
    qv: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    w: ArrayLike,
    hhl: ArrayLike,
    rotlat: ArrayLike,
    dlon: float,
    dlat: float,
    last_is_bottom: bool = True,
    radius: float = constants.R_earth,
) -> ArrayLike:
    r"""Compute the moisture flux convergence on a rotated lat/lon grid.

    Local-only helper for the MeteoSwiss fieldextra port, not meant for
    upstream. It follows fieldextra ``moisture_flux_convergence`` with the
    terrain-following metric terms of ``calc_sqrtg_r``, ``metric_coeffs``,
    ``gradient`` and ``divergence`` (horizontal branch).

    Parameters
    ----------
    qv: array-like
        Specific humidity on full levels, shape (nk, nj, ni) [:math:`kg kg^{-1}`].
        Rows run south to north, level 0 is the top.
    u: array-like
        Grid-relative u wind on full levels, staggered to i+1/2,
        shape (nk, nj, ni) [:math:`m s^{-1}`].
    v: array-like
        Grid-relative v wind on full levels, staggered to j+1/2,
        shape (nk, nj, ni) [:math:`m s^{-1}`].
    w: array-like
        Vertical wind on half levels, shape (nk+1, nj, ni) [:math:`m s^{-1}`].
    hhl: array-like
        Height of half levels, shape (nk+1, nj, ni) [m]; ``hhl[k]`` is the
        top of full level ``k``.
    rotlat: array-like
        Rotated latitude of each row, shape (nj,) [degrees].
    dlon: float
        Grid spacing along i [degrees].
    dlat: float
        Grid spacing along j [degrees].
    last_is_bottom: bool, optional
        If True, the last level is the lowest model level and uses a
        one-sided vertical derivative; otherwise it stays nan.
    radius: float, optional
        Earth radius [m], default :data:`earthkit.meteo.constants.R_earth`.

    Returns
    -------
    array-like
        Moisture flux convergence, shape (nk, nj, ni) [:math:`kg kg^{-1} s^{-1}`].
        Level 0 and the outer ring of cells are nan.

    Notes
    -----
    The result is

    .. math::

        MCONV = -q \nabla \cdot \mathbf{v} - \mathbf{v} \cdot \nabla q

    with the divergence and gradient taken in terrain-following
    coordinates on the rotated sphere.

    """
    xp = array_namespace(qv, u, v, w, hhl)
    qv = xp.asarray(qv)
    u = xp.asarray(u)
    v = xp.asarray(v)
    w = xp.asarray(w)
    hhl = xp.asarray(hhl)
    nk, nj, ni = qv.shape

    rad = xp.pi / 180.0
    inv_dlon = 1.0 / (dlon * rad)
    inv_dlat = 1.0 / (dlat * rad)
    lat = xp.asarray(rotlat, dtype=xp.float64, device=qv.device) * rad
    lat = lat[1 : nj - 1, None]
    ac = 1.0 / (xp.cos(lat) * radius)
    tg = xp.tan(lat)

    def c(a):
        return a[1 : nj - 1, 1 : ni - 1]

    def im(a):
        return a[1 : nj - 1, 0 : ni - 2]

    def ip(a):
        return a[1 : nj - 1, 2:ni]

    def jm(a):
        return a[0 : nj - 2, 1 : ni - 1]

    def jp(a):
        return a[2:nj, 1 : ni - 1]

    empty = xp.full((nj, ni), xp.nan, dtype=xp.float64, device=qv.device)
    levels = [empty]
    for k in range(1, nk):
        if k == nk - 1:
            if not last_is_bottom:
                levels.append(empty)
                continue
            centered, plus, minus = False, k, k - 1
        else:
            centered, plus, minus = True, k + 1, k - 1

        h, hp = hhl[k], hhl[k + 1]
        sqrtg_r = 1.0 / (c(h) - c(hp))
        dz_dlam = 0.25 * inv_dlon * sqrtg_r * ((ip(h) - im(h)) + (ip(hp) - im(hp)))
        dz_dphi = 0.25 * inv_dlat * sqrtg_r * ((jp(h) - jm(h)) + (jp(hp) - jm(hp)))
        wk_div = 0.25 if centered else 0.5
        wk_grad = 0.5 if centered else 1.0

        u0, up, um = u[k], u[plus], u[minus]
        v0, vp, vm = v[k], v[plus], v[minus]
        div_lam = (c(u0) - im(u0)) * inv_dlon + (
            (c(up) + im(up)) - (c(um) + im(um))
        ) * wk_div * dz_dlam
        div_phi = (
            (c(v0) - jm(v0)) * inv_dlat
            + ((c(vp) + jm(vp)) - (c(vm) + jm(vm))) * wk_div * dz_dphi
            + (c(w[k]) + c(w[k + 1]))
            - tg * 0.5 * (c(v0) + jm(v0))
        )
        div = ac * div_lam + (1.0 / radius) * div_phi

        q0, qp, qm = qv[k], qv[plus], qv[minus]
        grad1 = ac * (
            (ip(q0) - im(q0)) * 0.5 * inv_dlon + (c(qp) - c(qm)) * wk_grad * dz_dlam
        )
        grad2 = (1.0 / radius) * (
            (jp(q0) - jm(q0)) * 0.5 * inv_dlat + (c(qp) - c(qm)) * wk_grad * dz_dphi
        )
        res = -div * c(q0) - 0.5 * (grad1 * (im(u0) + c(u0)) + grad2 * (jm(v0) + c(v0)))
        levels.append(_pad_nan(xp, xp.astype(res, xp.float64)))
    return xp.stack(levels, axis=0)
