# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the local-only helpers of the fieldextra port (vertical)."""

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.vertical.array as vertical


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_ceiling_above_ground(xp, device):
    ceiling = xp.asarray([1500.25, np.nan, 3000.0], dtype=xp.float32, device=device)
    hsurf = xp.asarray([500.125, 400.0, 2999.5], dtype=xp.float64, device=device)

    r = vertical.ceiling_above_ground(ceiling, hsurf)
    assert r.dtype == xp.float32
    expected = xp.asarray([1000.125, np.nan, 0.5], dtype=xp.float32, device=device)
    assert xp.allclose(r, expected, equal_nan=True)
