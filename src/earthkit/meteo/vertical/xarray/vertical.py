# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array
from ..array.vertical import _geopotential_below_surface, _temperature_below_surface


def geopotential_height_from_geopotential(z: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_height_from_geopotential, z).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geopotential_height(gh: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        z = gh  g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geopotential_height, gh).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geopotential_height_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following formula:

    .. math::

        gh = \frac{h  R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geopotential_height_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)


    The computation is based on the following formula:

    .. math::

        z = \frac{h  g  R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geometric_height_from_geopotential_height(gh: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth}  gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential_height, gh, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


def geometric_height_from_geopotential(z: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential, z, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


def extrapolate_temperature_to_pressure_levels_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate temperature from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Temperature (K)

    See :func:`earthkit.meteo.vertical.array.extrapolate_temperature_to_pressure_levels_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(_temperature_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "air_temperature",
        "units": "K",
    })


def extrapolate_geopotential_to_pressure_levels_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate geopotential from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)

    See :func:`earthkit.meteo.vertical.array.extrapolate_geopotential_to_pressure_levels_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(_geopotential_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def extrapolate_below_lowest_level(
    data: xr.DataArray,
    z: xr.DataArray,
    target_z: xr.DataArray | float,
    mode: Literal["lapse_rate", "constant", "linear", "hydrostatic"] = "lapse_rate",
    lapse_rate: xr.DataArray | float = constants.standard_lapse_rate,
    vertical_dim: str = "z",
    start_z: xr.DataArray | float | None = None,
    start_value: xr.DataArray | float | None = None,
    t_start: xr.DataArray | float | None = None,
) -> xr.DataArray:
    r"""Extrapolate data to target heights below a start level, by default the lowest level.

    Parameters
    ----------
    data : xr.DataArray
        Data on levels, with the vertical dimension ``vertical_dim``.
    z : xr.DataArray
        Height of the levels. It can have only the vertical dimension.
    target_z : xr.DataArray | float
        Target heights, in the same units and with the same datum as ``z``. A float, a
        DataArray without extra dimension (one target per column), or a DataArray with one
        extra dimension, which is not a dimension of ``data`` and lists the target heights.
    mode : {"lapse_rate", "constant", "linear", "hydrostatic"}, default="lapse_rate"
        Extrapolation method.
    lapse_rate : xr.DataArray | float
        Lapse rate [K/m] for ``mode="lapse_rate"`` and ``mode="hydrostatic"``.
    vertical_dim : str, default="z"
        Name of the vertical dimension of ``data``, ``z`` and ``target_z``.
    start_z, start_value, t_start : xr.DataArray | float, optional
        Height, data and temperature of the start level, without ``vertical_dim``.

    Returns
    -------
    xr.DataArray
        The extrapolated data, without ``vertical_dim``. It has the extra dimension of
        ``target_z``, if there is one.

    See :func:`earthkit.meteo.vertical.array.extrapolate_below_lowest_level`
    for the definition.
    """
    squeeze = None
    if not isinstance(target_z, xr.DataArray):
        target_in, tcore, ocore = target_z, [], []
    else:
        extra = [d for d in target_z.dims if d not in data.dims]
        if len(extra) > 1:
            raise ValueError(f"target_z has more than one extra dimension: {extra}")
        if not extra:
            squeeze = "_target_level"
            target_z = target_z.expand_dims(squeeze)
            extra = [squeeze]
        ref = data.isel({vertical_dim: 0}, drop=True)
        target_in, tcore, ocore = target_z.broadcast_like(ref), extra, extra

    def _f(data, z, target, *rest):
        sz, sv, ts, lr = rest
        return array.extrapolate_below_lowest_level(
            data,
            z,
            target,
            mode=mode,
            lapse_rate=lr,
            vertical_dim=-1,
            start_z=sz,
            start_value=sv,
            t_start=ts,
        )

    out = xr.apply_ufunc(
        _f,
        data,
        z,
        target_in,
        start_z,
        start_value,
        t_start,
        lapse_rate,
        input_core_dims=[[vertical_dim], [vertical_dim], tcore, [], [], [], []],
        output_core_dims=[ocore],
    )
    return out if squeeze is None else out.squeeze(squeeze, drop=True)


def height_of_isotherm(
    t: xr.DataArray,
    z: xr.DataArray,
    t_target: float = constants.T_C2K,
    z_max: float | None = None,
    zs: xr.DataArray | None = None,
    extrapolate: bool = False,
    vertical_dim: str = "z",
) -> xr.DataArray:
    r"""Height where the temperature first reaches ``t_target`` when going down a column.

    Parameters
    ----------
    t : xr.DataArray
        Temperature on levels [K].
    z : xr.DataArray
        Height of the levels [m], monotonic along ``vertical_dim``. It can have only the
        vertical dimension.
    t_target : float, default=:data:`earthkit.meteo.constants.T_C2K`
        Temperature of the isotherm [K].
    z_max : float or None, default=None
        Height [m] above ``zs`` where the search starts. ``None`` starts at the top level.
    zs : xr.DataArray, optional
        Surface height [m].
    extrapolate : bool, default=False
        Extrapolate below the lowest level when no crossing is found.
    vertical_dim : str, default="z"
        Name of the vertical dimension of ``t`` and ``z``.

    Returns
    -------
    xr.DataArray
        Height of the isotherm [m], without ``vertical_dim``.

    See :func:`earthkit.meteo.vertical.array.height_of_isotherm`
    for the definition.
    """

    def _f(t, z, *zs_):
        return array.height_of_isotherm(
            t,
            z,
            t_target=t_target,
            z_max=z_max,
            zs=zs_[0] if zs_ else None,
            extrapolate=extrapolate,
            vertical_dim=-1,
        )

    args = [t, z] if zs is None else [t, z, zs]
    core = [[vertical_dim], [vertical_dim]] + ([] if zs is None else [[]])
    return xr.apply_ufunc(_f, *args, input_core_dims=core, output_core_dims=[[]]).assign_attrs({
        "units": "m",
    })


def nearest_level_index(
    coords: xr.DataArray,
    target_coords: xr.DataArray | float,
    max_diff: float | None = None,
    vertical_dim: str = "z",
) -> xr.DataArray:
    r"""Index of the level nearest to each target coordinate, per column.

    Parameters
    ----------
    coords : xr.DataArray
        Vertical coordinate of the levels (e.g. pressure or height). Levels where
        ``coords`` is nan are never selected.
    target_coords : xr.DataArray | float
        Target coordinates, same units as ``coords``. Broadcast against ``coords``
        following xarray rules; a DataArray with its own dimension (not ``vertical_dim``)
        adds that dimension to the result.
    max_diff : float, optional
        Largest accepted distance between a level and its target. When the nearest
        level is farther away the result is -1. ``None`` (default) accepts any distance.
    vertical_dim : str, default="z"
        Name of the vertical dimension of ``coords``.

    Returns
    -------
    xr.DataArray
        Integer level indices, without ``vertical_dim``.

    See :func:`earthkit.meteo.vertical.array.nearest_level_index`
    for the definition.
    """
    dist = abs(coords - target_coords).fillna(np.inf)
    dmin = dist.min(dim=vertical_dim)
    ok = dmin < np.inf
    if max_diff is not None:
        ok = ok & (dmin <= max_diff)
    return dist.argmin(dim=vertical_dim).where(ok, -1)
