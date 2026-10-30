# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Local-only helpers of the MeteoSwiss fieldextra port, not meant for upstream."""

from __future__ import annotations

from typing import Any

from earthkit.utils.array import array_namespace

ArrayLike = Any

__all__ = ["ceiling_above_ground"]


def ceiling_above_ground(ceiling: ArrayLike, hsurf: ArrayLike) -> ArrayLike:
    r"""Cloud ceiling above ground from the ceiling above mean sea level.

    .. math::

        c_{agl} = c_{amsl} - h_{surf}

    Local-only helper for the MeteoSwiss fieldextra port, not meant for upstream: it is a
    single subtraction.

    Parameters
    ----------
    ceiling : ArrayLike
        Cloud ceiling above mean sea level [m]. Nan means no ceiling.
    hsurf : ArrayLike
        Surface height above mean sea level [m].

    Returns
    -------
    ArrayLike
        Ceiling above ground [m], in the dtype of ``ceiling``: ``hsurf`` is cast to it before
        the subtraction, so float32 input is subtracted in float32. Nan where either input
        is nan.
    """
    xp = array_namespace(ceiling, hsurf)
    ceiling = xp.asarray(ceiling)
    return ceiling - xp.astype(xp.asarray(hsurf), ceiling.dtype)
