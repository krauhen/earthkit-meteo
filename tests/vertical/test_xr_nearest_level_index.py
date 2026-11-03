# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")

P = np.array([[50000.0, 61000.0], [70000.0, 84000.0], [85000.0, 92500.0]])


def _da(values=P, dims=("z", "cell")):
    import xarray as xr

    return xr.DataArray(values, dims=dims)


def test_xr_nearest_level_index():
    from earthkit.meteo.vertical.xarray import nearest_level_index

    res = nearest_level_index(_da(), 85000.0)

    assert res.dims == ("cell",)
    np.testing.assert_array_equal(res.values, [2, 1])


def test_xr_nearest_level_index_tie_and_max_diff():
    from earthkit.meteo.vertical.xarray import nearest_level_index

    np.testing.assert_array_equal(nearest_level_index(_da(), 77500.0).values, [1, 1])
    np.testing.assert_array_equal(nearest_level_index(_da(), 60000.0, max_diff=3000.0).values, [-1, 0])
    np.testing.assert_array_equal(nearest_level_index(_da(), np.nan).values, [-1, -1])


def test_xr_nearest_level_index_nan_level_and_targets():
    import xarray as xr

    from earthkit.meteo.vertical.xarray import nearest_level_index

    p = _da(np.array([[50000.0, 50000.0], [70000.0, np.nan], [85000.0, 85000.0]]))
    target = xr.DataArray([50000.0, 70000.0, 85000.0], dims="t")

    res = nearest_level_index(p, target)

    assert set(res.dims) == {"cell", "t"}
    np.testing.assert_array_equal(res.transpose("t", "cell").values, [[0, 0], [1, 2], [2, 2]])


def test_xr_nearest_level_index_vertical_dim():
    from earthkit.meteo.vertical.xarray import nearest_level_index

    res = nearest_level_index(_da(P.T, dims=("cell", "lev")), 85000.0, vertical_dim="lev")

    assert res.dims == ("cell",)
    np.testing.assert_array_equal(res.values, [2, 1])
