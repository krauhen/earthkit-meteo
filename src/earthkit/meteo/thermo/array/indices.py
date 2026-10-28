# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Convective and stability indices computed from vertical columns."""

from __future__ import annotations

from typing import Any

from earthkit.utils.array import array_namespace

from earthkit.meteo import constants

ArrayLike = Any


def _levels_first(xp, a, vertical_dim):
    return a if vertical_dim == 0 else xp.moveaxis(a, vertical_dim, 0)


def bulk_richardson_number(
    theta_v: ArrayLike,
    z: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    z_sfc: ArrayLike = 0.0,
    vertical_dim: int = 0,
    vmin: float = 1.0e-3,
) -> ArrayLike:
    r"""Compute the bulk Richardson number on each level relative to the lowest level.

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
    vertical_dim: int, optional
        Vertical axis of ``theta_v``, ``z``, ``u`` and ``v`` (default: 0).
    vmin: float, optional
        Lower bound of the wind speed (m/s) in the denominator (default: 1e-3).

    Returns
    -------
    array-like
        Bulk Richardson number (dimensionless) on each level.


    The levels must be ordered from the top down: the last level along ``vertical_dim``
    is the lowest one, :math:`n`. The computation is based on the following formula
    (Stull, 1988):

    .. math::

        Ri_{b,k} = \frac{g \, (z_k - z_{sfc}) \, (\theta_{v,k} - \theta_{v,n})}
        {\overline{\theta_v}_k \, \max(u_k^2 + v_k^2, v_{min}^2)}

    where:

        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)
        * :math:`\overline{\theta_v}_k` is the plain mean of :math:`\theta_v` over the
          levels :math:`k` to :math:`n`; a nan on any of them makes level :math:`k` nan

    The value on the lowest level is 0.

    References
    ----------
    Stull, R. B., 1988: An Introduction to Boundary Layer Meteorology. Kluwer, 666 pp.
    """
    xp = array_namespace(theta_v, z, u, v)
    theta_v = _levels_first(xp, xp.asarray(theta_v), vertical_dim)
    z = _levels_first(xp, xp.asarray(z), vertical_dim)
    u = _levels_first(xp, xp.asarray(u), vertical_dim)
    v = _levels_first(xp, xp.asarray(v), vertical_dim)
    z_sfc = xp.asarray(z_sfc)

    nlev = theta_v.shape[0]
    total = xp.flip(xp.cumulative_sum(xp.flip(theta_v, axis=0), axis=0), axis=0)
    count = xp.arange(nlev, 0, -1, dtype=theta_v.dtype)
    count = xp.reshape(count, (nlev,) + (1,) * (theta_v.ndim - 1))
    mean = total / count

    speed2 = xp.maximum(u**2 + v**2, xp.asarray(vmin**2, dtype=u.dtype))
    ri = constants.g * (z - z_sfc) * (theta_v - theta_v[-1, ...]) / (mean * speed2)
    return ri if vertical_dim == 0 else xp.moveaxis(ri, 0, vertical_dim)
