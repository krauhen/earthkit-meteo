# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.wind as wind_top
import earthkit.meteo.wind.array as wind
from earthkit.meteo import constants


def _grid(nk=4, nj=7, ni=8, d=0.1):
    rotlat = (np.arange(nj) - nj // 2) * d
    rotlon = (np.arange(ni) - ni // 2) * d
    hhl = np.broadcast_to(np.linspace(3000.0, 0.0, nk + 1)[:, None, None], (nk + 1, nj, ni))
    return rotlat, rotlon, np.ascontiguousarray(hhl), d


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_moisture_flux_convergence_uniform(xp, device):
    rotlat, _, hhl, d = _grid()
    nk, nj, ni = hhl.shape[0] - 1, hhl.shape[1], hhl.shape[2]
    qv = xp.asarray(np.full((nk, nj, ni), 0.005), device=device)
    u = xp.asarray(np.full((nk, nj, ni), 10.0), device=device)
    v = xp.asarray(np.zeros((nk, nj, ni)), device=device)
    w = xp.asarray(np.zeros((nk + 1, nj, ni)), device=device)

    r = wind.moisture_flux_convergence(qv, u, v, w, xp.asarray(hhl, device=device), rotlat, d, d)
    assert r.shape == (nk, nj, ni)
    assert bool(xp.all(xp.isnan(r[0])))
    assert bool(xp.all(xp.isnan(r[:, 0, :]))) and bool(xp.all(xp.isnan(r[:, :, -1])))
    assert xp.allclose(r[1:, 1:-1, 1:-1], xp.zeros_like(r[1:, 1:-1, 1:-1]), atol=1e-18)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_moisture_flux_convergence_radial_inflow(xp, device):
    rotlat, rotlon, hhl, d = _grid()
    nk, nj, ni = hhl.shape[0] - 1, hhl.shape[1], hhl.shape[2]
    rad = np.pi / 180.0
    a, q0, radius = 1e-5, 0.005, constants.R_earth
    lat = rotlat[:, None] * rad
    # u at i+1/2 and v at j+1/2, flowing towards the grid centre: div = -2a
    u2 = -a * radius * np.cos(lat) * (rotlon[None, :] + 0.5 * d) * rad
    v2 = -a * radius * (rotlat[:, None] + 0.5 * d) * rad + 0.0 * rotlon[None, :]
    u = xp.asarray(np.broadcast_to(u2, (nk, nj, ni)).copy(), device=device)
    v = xp.asarray(np.broadcast_to(v2, (nk, nj, ni)).copy(), device=device)
    qv = xp.asarray(np.full((nk, nj, ni), q0), device=device)
    w = xp.asarray(np.zeros((nk + 1, nj, ni)), device=device)

    r = wind_top.moisture_flux_convergence(qv, u, v, w, xp.asarray(hhl, device=device), rotlat, d, d)
    inner = r[1:, 1:-1, 1:-1]
    expected = 2.0 * a * q0
    assert bool(xp.all(inner > 0))
    assert xp.allclose(inner, xp.full_like(inner, expected), rtol=0.01)


def test_moisture_flux_convergence_last_level():
    rotlat, _, hhl, d = _grid()
    nk, nj, ni = hhl.shape[0] - 1, hhl.shape[1], hhl.shape[2]
    rng = np.random.default_rng(0)
    qv = rng.uniform(0.001, 0.01, (nk, nj, ni))
    u, v = rng.normal(0, 5, (2, nk, nj, ni))
    w = rng.normal(0, 0.1, (nk + 1, nj, ni))
    r = wind.moisture_flux_convergence(qv, u, v, w, hhl, rotlat, d, d, last_is_bottom=False)
    assert np.all(np.isnan(r[-1]))
    assert np.all(np.isfinite(r[1:-1, 1:-1, 1:-1]))
