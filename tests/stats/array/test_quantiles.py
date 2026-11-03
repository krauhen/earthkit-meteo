# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os
import sys

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import stats

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def _get_quantile_data():
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from _quantile import q_test_array

    return q_test_array


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("method", ["sort", "numpy_bulk", "numpy"])
@pytest.mark.parametrize(
    "data,which,kwargs,v_ref",
    [
        (
            [
                [3, 6, 12, 0, 45, 0, 4, 0, 7],
                [1, 2.5, 4, 3, 18, 3, 8, 7, 2],
                [3, 4, 9, 0.5, 2, 48, 4, 9, 6],
                [5, 19, 8.3, 2, 6, 1, 0, 1, 0],
            ],
            4,
            dict(dim=1),
            [
                [0, 1, 0.5, 0],
                [0, 2.5, 3, 1],
                [4, 3, 4, 2],
                [7, 7, 9, 6],
                [45, 18, 48, 19],
            ],
        ),
        (
            [
                [3, 6, 12, 0, 45, 0, 4, 0, 7],
                [1, 2.5, 4, 3, 18, 3, 8, 7, 2],
                [3, 4, 9, 0.5, 2, 48, 4, 9, 6],
                [5, 19, 8.3, 2, 6, 1, 0, 1, 0],
            ],
            [0.5, 1.0],
            {},
            # TODO: check if the commented or uncommented data block below is the
            # correct reference data. The commented data block is the one that
            # was present in the original test, but the results were not tested
            # against it, so it is not clear if it is correct.
            # [
            #     [3, 5, 6.15, 1.25, 12, 2, 4, 4, 4],
            #     [5, 19, 12, 3, 45, 48, 8, 9, 7],
            # ],
            [
                [
                    3.0000000000,
                    5.0000000000,
                    8.6500000000,
                    1.2500000000,
                    12.0000000000,
                    2.0000000000,
                    4.0000000000,
                    4.0000000000,
                    4.0000000000,
                ],
                [
                    5.0000000000,
                    19.0000000000,
                    12.0000000000,
                    3.0000000000,
                    45.0000000000,
                    48.0000000000,
                    8.0000000000,
                    9.0000000000,
                    7.0000000000,
                ],
            ],
        ),
    ],
)
def test_quantiles_core(xp, device, data, which, kwargs, v_ref, method):
    data, v_ref = xp.asarray(data, device=device), xp.asarray(v_ref, device=device)

    r = list(stats.iter_quantiles(data, which, method=method, **kwargs))
    assert len(r) == v_ref.shape[0]
    for i, d in enumerate(r):
        # this is needed to handle the case where the last dimension is 1
        if d.ndim >= 2 and d.shape[-1] == 1:
            d = d[..., 0]

        assert xp.allclose(d, v_ref[i], rtol=1e-4), f"i={i}, d={d}, v_ref={v_ref[i]}"


# TODO: reimplement this test to use reference values
# TODO: this! test fails with cupy. The reason is that cupy.quantile works differently
#       than np.quantile when nans are present
@pytest.mark.parametrize(
    "xp, device",
    list(filter(lambda x: x[0]._earthkit_array_namespace_name != "cupy", NAMESPACE_DEVICES)),
)
@pytest.mark.parametrize("arr", [_get_quantile_data()])
def test_quantiles_nans(xp, device, arr):
    arr = xp.asarray(arr, device=device)
    qs = xp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    bulk = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy_bulk")])
    r1 = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="sort")])
    assert bulk.shape == r1.shape
    assert xp.allclose(bulk, r1, equal_nan=True)
    r2 = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy")])
    assert bulk.shape == r2.shape
    assert xp.allclose(bulk, r2, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_quantiles_dim_argument(xp, device):
    arr = xp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], device=device)
    got = xp.asarray([q for q in stats.iter_quantiles(arr, which=[0.5], dim=1, method="numpy")])[0]
    ref = xp.asarray([2.0, 4.0], device=device)
    assert xp.allclose(got, ref)


def _hazen_reference(members, q_fraction):
    """Cell-by-cell float32 loop of fieldextra's equal-weight quantile (fxtr_operator_probability.f90:1318-1338)."""
    f32 = np.float32
    members = np.asarray(members, dtype=f32)
    out = np.full(members.shape[1:], np.nan, dtype=f32)
    q = f32(q_fraction)
    n = members.shape[0]
    # the Fortran statement as written: cum(i) = cum(i-1) + 0.5 * (1/sum_weight) * (w(i-1) + w(i))
    w = f32(1.0) / f32(n)
    half = f32(0.5)
    cum = [half * w]
    for _ in range(1, n):
        cum.append(cum[-1] + half * (w + w))
    for idx in np.ndindex(*members.shape[1:]):
        col = members[(slice(None),) + idx]
        if not np.all(np.isfinite(col)):
            continue
        col = np.sort(col)
        value = col[-1]
        for i in range(n):
            if cum[i] > q:
                if i == 0:
                    value = col[0]
                else:
                    value = col[i - 1] + (col[i] - col[i - 1]) * ((q - cum[i - 1]) / (cum[i] - cum[i - 1]))
                break
        out[idx] = value
    return out


def _eps_quantile(members, quantile=50.0):
    # the contract of the local eps_quantile copies in the ported products:
    # (members on axis 0, quantile in percent) -> float32, NaN where any member is not finite
    (result,) = stats.iter_quantiles(members, [quantile / 100.0], dim=0, method="hazen")
    return result


def _members(n, shape=(4, 3), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n,) + shape) * 30.0 - 5.0).astype(np.float32)


def test_hazen_plotting_positions():
    # 4 members: positions 0.125, 0.375, 0.625, 0.875
    m = np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float32)
    got = [float(next(stats.iter_quantiles(m, [q], method="hazen"))) for q in (0.0, 0.125, 0.25, 0.5, 0.875, 1.0)]
    assert got == [1.0, 1.0, 1.5, 2.5, 4.0, 4.0]


@pytest.mark.parametrize("n", [2, 3, 11, 21, 51])
@pytest.mark.parametrize("quantile", [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0])
def test_hazen_matches_fieldextra_rule_exactly(n, quantile):
    m = _members(n, seed=n)
    got = _eps_quantile(m, quantile)
    ref = _hazen_reference(m, np.float32(quantile / 100.0))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hazen_is_float32(dtype):
    m = _members(11).astype(dtype)
    got = _eps_quantile(m, 75.0)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, _hazen_reference(m, np.float32(0.75)))


def test_hazen_nan_when_any_member_not_finite():
    m = _members(11)
    m[3, 1, 2] = np.nan
    m[7, 0, 0] = np.inf
    m[0, 3, 1] = -np.inf
    got = _eps_quantile(m, 50.0)
    bad = ~np.all(np.isfinite(m), axis=0)
    assert bad.sum() == 3
    assert np.all(np.isnan(got[bad]))
    assert np.all(np.isfinite(got[~bad]))
    np.testing.assert_array_equal(got, _hazen_reference(m, np.float32(0.5)))


def test_hazen_dim_and_which():
    m = _members(11)
    moved = np.moveaxis(m, 0, 2)
    r0 = list(stats.iter_quantiles(m, [0.1, 0.5, 0.9], dim=0, method="hazen"))
    r2 = list(stats.iter_quantiles(moved, [0.1, 0.5, 0.9], dim=2, method="hazen"))
    assert len(r0) == len(r2) == 3
    for a, b in zip(r0, r2):
        np.testing.assert_array_equal(a, b)
    # an int is a number of intervals, as for the other methods: 0, 25, 50, 75, 100 %
    r4 = list(stats.iter_quantiles(m, 4, method="hazen"))
    assert len(r4) == 5
    for a, q in zip(r4, (0.0, 0.25, 0.5, 0.75, 1.0)):
        np.testing.assert_array_equal(a, _hazen_reference(m, np.float32(q)))


def test_hazen_extremes_outside_the_positions():
    m = _members(5)
    np.testing.assert_array_equal(_eps_quantile(m, 0.0), m.min(axis=0))
    np.testing.assert_array_equal(_eps_quantile(m, 100.0), m.max(axis=0))


def test_hazen_numpy_from_dataarray():
    # iter_quantiles has no xarray layer upstream: the DataArray's numpy data is the input
    xr = pytest.importorskip("xarray")
    m = _members(11)
    da = xr.DataArray(m, dims=("member", "y", "x"))
    got = _eps_quantile(da.data, 25.0)
    np.testing.assert_array_equal(got, _hazen_reference(da.values, np.float32(0.25)))


def test_hazen_fixed_values_pin_the_single_rounding_form():
    # hard-coded float32 results where (cum + step) + step and the single-rounding form differ
    v11 = np.array([0.3, 1.7, 2.9, 0.05, 4.4, 3.1, 0.9, 2.2, 5.6, 1.1, 3.8], dtype=np.float32)
    v21 = np.array(
        [0.0, 0.37, 1.48, 3.33, 5.92, 1.95, 6.02, 3.53, 1.78, 0.77, 0.5,
         0.97, 2.18, 4.13, 6.82, 2.95, 7.12, 4.73, 3.08, 2.17, 2.0],
        dtype=np.float32,
    )  # fmt: skip
    assert float(_eps_quantile(v11[:, None], 75.0)[0]) == 3.6249992847442627
    assert float(_eps_quantile(v21[:, None], 90.0)[0]) == 6.340000152587891


def test_hazen_invalid_method_message():
    with pytest.raises(ValueError, match="hazen"):
        list(stats.iter_quantiles(_members(3), [0.5], method="unknown"))
