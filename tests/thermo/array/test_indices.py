# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the convective and stability indices."""
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.thermo.array  # noqa: F401 (makes thermo.array available)
from earthkit.meteo import constants
from earthkit.meteo import thermo


@pytest.fixture(params=["array", "top"])
def mod(request):
    return thermo.array if request.param == "array" else thermo


_RI_Z = [3000.0, 2000.0, 1000.0, 500.0, 100.0]


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_bulk_richardson_number_neutral(mod, xp, device):
    # constant theta_v: zero everywhere even with strong shear
    theta_v = xp.full((5, 2), 300.0, device=device)
    z = xp.asarray([[h, h] for h in _RI_Z], device=device)
    u = xp.asarray([[20.0, 5.0], [15.0, 4.0], [10.0, 3.0], [5.0, 2.0], [1.0, 1.0]], device=device)
    v = xp.zeros_like(u)

    r = mod.bulk_richardson_number(theta_v, z, u, v)
    assert r.shape == (5, 2)
    assert xp.all(r == 0.0)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_bulk_richardson_number_stable(mod, xp, device):
    theta_v = xp.asarray([310.0, 306.0, 303.0, 301.0, 300.0], device=device)
    z = xp.asarray(_RI_Z, device=device)
    u = xp.asarray([10.0, 8.0, 6.0, 4.0, 2.0], device=device)
    v = xp.asarray([0.0, 6.0, 0.0, 3.0, 0.0], device=device)

    r = mod.bulk_richardson_number(theta_v, z, u, v, z_sfc=50.0)
    mean = (306.0 + 303.0 + 301.0 + 300.0) / 4.0
    expected = constants.g * (2000.0 - 50.0) * 6.0 / (mean * 100.0)
    assert xp.all(r[:-1] > 0.0)
    assert r[-1] == 0.0
    assert abs(float(r[1]) - expected) < 1e-12


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_bulk_richardson_number_vmin(mod, xp, device):
    # calm wind on level 0: the speed is floored at vmin; levels on the last axis
    theta_v = xp.asarray([[301.0, 300.0]], device=device)
    z = xp.asarray([[1000.0, 0.0]], device=device)
    u = xp.zeros_like(theta_v)

    r = mod.bulk_richardson_number(theta_v, z, u, u, vertical_dim=1, vmin=0.5)
    expected = constants.g * 1000.0 * 1.0 / (300.5 * 0.25)
    assert r.shape == (1, 2)
    assert abs(float(r[0, 0]) - expected) < 1e-9
