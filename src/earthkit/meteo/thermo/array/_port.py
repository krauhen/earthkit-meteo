# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Local-only helpers of the MeteoSwiss fieldextra port, not meant for upstream.

``pbl_height_bulk_richardson`` is only exported by :mod:`earthkit.meteo.thermo`, where it
shadows the upstream function of the same name. It is not exported here, because the
upstream high-level dispatch looks its implementation up in this namespace.
"""

from __future__ import annotations

from typing import Any

from earthkit.utils.array import array_namespace

from earthkit.meteo.vertical.array.vertical import _levels_first

from .thermo import bulk_richardson_number, wet_bulb_temperature_from_specific_humidity

ArrayLike = Any

__all__ = ["wet_bulb_temperature_cosmo"]


def wet_bulb_temperature_cosmo(t: ArrayLike, q: ArrayLike, p: ArrayLike) -> ArrayLike:
    r"""Wet bulb temperature (K) with the closed-form COSMO approximation.

    Same as :func:`wet_bulb_temperature_from_specific_humidity` with ``method="cosmo"``.
    """
    return wet_bulb_temperature_from_specific_humidity(t, q, p, method="cosmo")


def pbl_height_bulk_richardson(
    theta_v: ArrayLike,
    z: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    z_sfc: ArrayLike = 0.0,
    ri_crit: float = 0.22,
    max_height: float = 5000.0,
    vertical_dim: int = 0,
) -> ArrayLike:
    r"""Compute the boundary layer height from the bulk Richardson number.

    .. note::

        This follows the COSMO/ICON post-processing tool fieldextra (``pbl_height``),
        including its field-wide fallback level described below. That fallback depends on
        all points passed in together, so the result at one point can change with the
        extent of the input. It is meant for reproducing fieldextra output.

    Parameters
    ----------
    theta_v: array-like
        Virtual potential temperature on levels (K).
    z: array-like
        Height of the levels above sea level (m).
    u: array-like
        Eastward wind component on levels (m/s).
    v: array-like
        Northward wind component on levels (m/s).
    z_sfc: array-like, optional
        Surface height above sea level (m), shaped like one level (default: 0).
    ri_crit: float, optional
        Critical bulk Richardson number (default: 0.22).
    max_height: float, optional
        Height above ground (m) the fallback level must reach at every point
        (default: 5000).
    vertical_dim: int, optional
        Vertical axis of ``theta_v``, ``z``, ``u`` and ``v`` (default: 0).

    Returns
    -------
    array-like
        Boundary layer height above ground (m), without the vertical axis.


    The bulk Richardson number :math:`Ri_b` comes from :func:`bulk_richardson_number`,
    so the levels must be ordered from the top down. The fallback level :math:`k_{top}` is
    the lowest level whose height above ground :math:`z_k - z_{sfc}` is at least
    ``max_height`` at every point; it is one index for the whole input. Going up from
    the lowest level to :math:`k_{top}`, the height is that of the first level with

    .. math::

        Ri_{b,k} > Ri_{crit}

    given as :math:`z_k - z_{sfc}`, without interpolation between levels. Where no level
    up to :math:`k_{top}` passes, the height is :math:`z_{k_{top}} - z_{sfc}`. A point is
    nan where :math:`z`, :math:`z_{sfc}` or :math:`Ri_b` is nan on any level from
    :math:`k_{top}` down.

    Raises
    ------
    ValueError
        If no level reaches ``max_height`` above ground at every point.

    References
    ----------
    Vogelezang, D. H. P., and A. A. M. Holtslag, 1996: Evaluation and model impacts of
    alternative boundary-layer height formulations. Boundary-Layer Meteorol., 81, 245-269.
    """
    xp = array_namespace(theta_v, z, u, v)
    theta_v = _levels_first(xp, xp.asarray(theta_v), vertical_dim)
    z = _levels_first(xp, xp.asarray(z), vertical_dim)
    u = _levels_first(xp, xp.asarray(u), vertical_dim)
    v = _levels_first(xp, xp.asarray(v), vertical_dim)
    z_sfc = xp.asarray(z_sfc)

    ri = bulk_richardson_number(theta_v, z, u, v, z_sfc=z_sfc)
    h = z - z_sfc

    top = None
    for k in range(h.shape[0] - 1, -1, -1):
        if bool(xp.min(h[k, ...]) >= max_height):
            top = k
            break
    if top is None:
        raise ValueError(f"no level reaches {max_height} m above ground at every point")

    ri = ri[top:, ...]
    h = h[top:, ...]
    ok = xp.all(xp.isfinite(h), axis=0) & xp.all(xp.isfinite(ri), axis=0)

    hit = xp.astype(ri > ri_crit, xp.int8)
    found = xp.any(hit > 0, axis=0)
    first = h.shape[0] - 1 - xp.argmax(xp.flip(hit, axis=0), axis=0)
    h_hit = xp.take_along_axis(h, first[xp.newaxis, ...], axis=0)[0]

    out = xp.where(found, h_hit, h[0, ...])
    return xp.where(ok, out, xp.asarray(xp.nan, dtype=out.dtype))
