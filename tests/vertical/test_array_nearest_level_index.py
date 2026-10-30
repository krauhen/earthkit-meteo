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

from earthkit.meteo.vertical.array import nearest_level_index

P = np.array([[50000.0, 61000.0], [70000.0, 84000.0], [85000.0, 92500.0]])


def test_array_nearest_level_index_scalar_target():
    res = nearest_level_index(P, 85000.0)

    assert res.shape == (2,)
    np.testing.assert_array_equal(res, [2, 1])


def test_array_nearest_level_index_tie_goes_to_lower_index():
    res = nearest_level_index(P, 77500.0)

    np.testing.assert_array_equal(res, [1, 1])


def test_array_nearest_level_index_max_diff():
    res = nearest_level_index(P, 60000.0, max_diff=3000.0)

    np.testing.assert_array_equal(res, [-1, 0])


def test_array_nearest_level_index_nan_target():
    res = nearest_level_index(P, np.nan)

    np.testing.assert_array_equal(res, [-1, -1])


def test_array_nearest_level_index_nan_level_is_skipped():
    p = np.array([[50000.0, 70000.0, 85000.0], [50000.0, np.nan, 85000.0]])
    target = np.array([50000.0, 70000.0, 85000.0])

    res = nearest_level_index(p, target, vertical_dim=1)

    assert res.shape == (2, 3)
    np.testing.assert_array_equal(res, [[0, 1, 2], [0, 2, 2]])


def test_array_nearest_level_index_all_nan_column():
    p = np.array([[np.nan, 1.0], [np.nan, 2.0]])

    res = nearest_level_index(p, 1.0)

    np.testing.assert_array_equal(res, [-1, 0])


def test_array_nearest_level_index_single_level():
    res = nearest_level_index(np.array([[1000.0, 2000.0]]), 1500.0, max_diff=100.0)

    np.testing.assert_array_equal(res, [-1, -1])


def test_array_nearest_level_index_target_with_levels_axis():
    target = np.array([[50000.0, 61000.0], [85000.0, 92500.0]])

    res = nearest_level_index(P, target)

    assert res.shape == (2, 2)
    np.testing.assert_array_equal(res, [[0, 0], [2, 2]])


def test_array_nearest_level_index_invalid_target_shape():
    with pytest.raises(ValueError, match="does not match one level"):
        nearest_level_index(P, np.zeros((2, 3)))
