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

from .array._port import pbl_height_bulk_richardson
from .thermo import wet_bulb_temperature_from_specific_humidity

__all__ = ["pbl_height_bulk_richardson", "wet_bulb_temperature_cosmo"]


def wet_bulb_temperature_cosmo(t, q, p):
    r"""Wet bulb temperature (K) with the closed-form COSMO approximation.

    Same as :func:`wet_bulb_temperature_from_specific_humidity` with ``method="cosmo"``.
    """
    return wet_bulb_temperature_from_specific_humidity(t, q, p, method="cosmo")
