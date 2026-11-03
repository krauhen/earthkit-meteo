# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the local-only helpers of the fieldextra port (thermo)."""

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.thermo.array  # noqa: F401 (makes thermo.array available)
from earthkit.meteo import thermo


@pytest.fixture(params=["array", "top"])
def mod(request):
    return thermo.array if request.param == "array" else thermo


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_wet_bulb_temperature_cosmo_alias(mod, xp, device):
    t = xp.asarray([263.15, 293.15], device=device)
    q = xp.asarray([0.001, 0.008], device=device)
    p = xp.asarray([85000.0, 100000.0], device=device)
    got = mod.wet_bulb_temperature_cosmo(t, q, p)
    ref = thermo.array.wet_bulb_temperature_from_specific_humidity(t, q, p, method="cosmo")
    assert xp.all(got == ref)


_PBL_Z = [6000.0, 4000.0, 2000.0, 1000.0, 500.0, 10.0]


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pbl_height_bulk_richardson(xp, device):
    # column 0 is stable from 1000 m up; column 1 is neutral and takes the fallback
    theta_v = xp.asarray(
        [[310.0, 300.0], [306.0, 300.0], [304.0, 300.0], [302.0, 300.0], [300.0, 300.0], [300.0, 300.0]],
        device=device,
    )
    z = xp.asarray([[h, h] for h in _PBL_Z], device=device)
    u = xp.full((6, 2), 5.0, device=device)
    v = xp.zeros_like(u)

    r = thermo.pbl_height_bulk_richardson(theta_v, z, u, v, z_sfc=10.0)
    assert r.shape == (2,)
    assert xp.all(r == xp.asarray([990.0, 5990.0], device=device))

    # a higher critical value moves column 0 up one level; levels on the last axis
    ri = thermo.bulk_richardson_number(theta_v, z, u, v, z_sfc=10.0)
    r = thermo.pbl_height_bulk_richardson(theta_v.T, z.T, u.T, v.T, 10.0, ri_crit=float(ri[3, 0]), vertical_dim=1)
    assert xp.all(r == xp.asarray([1990.0, 5990.0], device=device))


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pbl_height_bulk_richardson_nan_and_error(xp, device):
    theta_v = xp.asarray([[310.0, 310.0], [300.0, np.nan], [300.0, 300.0]], device=device)
    z = xp.asarray([[6000.0, 6000.0], [1000.0, 1000.0], [10.0, 10.0]], device=device)
    u = xp.full((3, 2), 5.0, device=device)

    r = thermo.pbl_height_bulk_richardson(theta_v, z, u, u)
    assert float(r[0]) == 6000.0
    assert xp.isnan(r[1])

    with pytest.raises(ValueError):
        thermo.pbl_height_bulk_richardson(theta_v, z, u, u, max_height=7000.0)
