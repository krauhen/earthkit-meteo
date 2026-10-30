# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Shared test bodies for the array-level and high-level vertical APIs.

These functions are parametrized by the ``vertical`` fixture, which each
concrete test module (test_array_vertical.py, test_vertical.py) defines to
point at the module under test (``earthkit.meteo.vertical.array`` or
``earthkit.meteo.vertical``). This file is not collected directly by pytest
(it does not match the ``test_*.py`` pattern); the concrete test modules
import these functions with ``from _vertical_cases import *``.
"""

from importlib import import_module

import numpy as np
import pytest
from earthkit.utils.array.namespace import _NUMPY_NAMESPACE
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo.utils.testing import Tolerance

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})


def _get_data(name):
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    return import_module(name)


DATA_HYBRID_CORE = _get_data("_hybrid_core_data")
DATA_HYBRID_H = _get_data("_hybrid_height_data")


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "z,expected_value",
    [
        (0.0, 0.0),
        (1000.0, 101.97162129779284),
        ([1000.0, 10000.0], [101.9716212978, 1019.7162129779]),
    ],
)
def test_geopotential_height_from_geopotential(vertical, z, expected_value, xp, device):
    z = xp.asarray(z, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_height_from_geopotential(z)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (101.97162129779284, 1000.0),
        ([101.9716212978, 1019.7162129779], [1000.0, 10000.0]),
    ],
)
def test_geopotential_from_geopotential_height(vertical, h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_from_geopotential_height(h)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5102.664476187331, 50000.0),
        (
            [1019.8794448450, 5102.6644761873, 7146.0195417809],
            [10000.0, 50000.0, 70000.0],
        ),
    ],
)
def test_geopotential_from_geometric_height(vertical, h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_from_geometric_height(h)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5003.9269715243, 5000.0),
        ([1000.1569802279, 5003.9269715243, 7007.6992829768], [1000.0, 5000.0, 7000.0]),
    ],
)
def test_geopotential_height_from_geometric_height(vertical, h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_height_from_geometric_height(h)

    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "z,expected_value",
    [
        (0.0, 0.0),
        (50000.0, 5102.664476187331),
        (
            [10000.0, 50000.0, 70000.0],
            [1019.8794448450, 5102.6644761873, 7146.0195417809],
        ),
    ],
)
def test_geometric_height_from_geopotential(vertical, z, expected_value, xp, device):
    z = xp.asarray(z, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geometric_height_from_geopotential(z)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "zh,expected_value",
    [
        (0.0, 0.0),
        (5000.0, 5003.9269715243),
        ([1000.0, 5000.0, 7000.0], [1000.1569802279, 5003.9269715243, 7007.6992829768]),
    ],
)
def test_geometric_height_from_geopotential_height(vertical, zh, expected_value, xp, device):
    zh = xp.asarray(zh, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geometric_height_from_geopotential_height(zh)
    assert xp.allclose(r, expected_value)


# NOTE: this method returns numpy arrays only
@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
def test_hybrid_level_parameters_1(vertical, xp):
    ref_A = DATA_HYBRID_CORE.A
    ref_B = DATA_HYBRID_CORE.B
    ref_A, ref_B = (xp.asarray(x) for x in [ref_A, ref_B])

    A, B = vertical.hybrid_level_parameters(137)

    # Note: A in test data has been stored with higher precision than in the conf
    assert np.allclose(A, ref_A, rtol=1e-5)
    assert np.allclose(B, ref_B, rtol=1e-5)


def test_hybrid_level_parameters_2(vertical):
    with pytest.raises(ValueError):
        vertical.hybrid_level_parameters(-100)


def test_hybrid_level_parameters_3(vertical):
    with pytest.raises(ValueError):
        vertical.hybrid_level_parameters(137, model="unknown_model")


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_pressure_on_hybrid_levels_core(vertical, index, xp, device):
    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x, device=device) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    sp = sp[index[1]]
    ref_p_full = ref_p_full[index]
    ref_p_half = ref_p_half[index]
    ref_delta = ref_delta[index]
    ref_alpha = ref_alpha[index]

    p_full, p_half, delta, alpha, res_levels = vertical.pressure_on_hybrid_levels(
        sp, A, B, alpha_top="ifs", output=["full", "half", "delta", "alpha", "level"]
    )

    assert xp.allclose(res_levels["full"], xp.arange(1, len(A), dtype=int))
    assert xp.allclose(res_levels["half"], xp.arange(0, len(A), dtype=int))

    tolerance = Tolerance({
        "p_full": {64: (1e-8, 1e-6)},
        "p_half": {64: (1e-8, 1e-6)},
        "delta": {64: (1e-8, 1e-6), 32: (1e-6, 1e-5)},
        "alpha": {64: (1e-8, 1e-6), 32: (1e-4, 1e-5)},
    })
    atol, rtol = tolerance.get(key="p_full", dtype=sp.dtype)
    assert xp.allclose(p_full, ref_p_full, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="p_half", dtype=sp.dtype)
    assert xp.allclose(p_half, ref_p_half, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="delta", dtype=sp.dtype)
    assert xp.allclose(delta, ref_delta, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="alpha", dtype=sp.dtype)
    assert xp.allclose(alpha, ref_alpha, atol=atol, rtol=rtol)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_pressure_on_hybrid_levels_axis(vertical, index, xp, device):
    # nondefault vertical axis for output
    vertical_dim = 1

    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

    nlev = len(A) - 1

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x, device=device) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    sp = sp[index[1]]
    ref_p_full = ref_p_full[index]
    ref_p_half = ref_p_half[index]
    ref_delta = ref_delta[index]
    ref_alpha = ref_alpha[index]

    p_full, p_half, delta, alpha, res_levels = vertical.pressure_on_hybrid_levels(
        sp,
        A,
        B,
        alpha_top="ifs",
        output=["full", "half", "delta", "alpha", "level"],
        vertical_dim=vertical_dim,
    )

    assert xp.allclose(res_levels["full"], xp.arange(1, len(A), dtype=int))
    assert xp.allclose(res_levels["half"], xp.arange(0, len(A), dtype=int))

    input_shape = sp.shape

    assert p_full.shape == input_shape + (nlev,)
    assert p_half.shape == input_shape + (nlev + 1,)
    assert delta.shape == input_shape + (nlev,)
    assert alpha.shape == input_shape + (nlev,)

    if p_full.ndim > 1:
        p_full = xp.moveaxis(p_full, vertical_dim, 0)
        p_half = xp.moveaxis(p_half, vertical_dim, 0)
        delta = xp.moveaxis(delta, vertical_dim, 0)
        alpha = xp.moveaxis(alpha, vertical_dim, 0)

    tolerance = Tolerance({
        "p_full": {64: (1e-8, 1e-6)},
        "p_half": {64: (1e-8, 1e-6)},
        "delta": {64: (1e-8, 1e-6), 32: (1e-6, 1e-5)},
        "alpha": {64: (1e-8, 1e-6), 32: (1e-4, 1e-5)},
    })
    atol, rtol = tolerance.get(key="p_full", dtype=sp.dtype)
    assert xp.allclose(p_full, ref_p_full, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="p_half", dtype=sp.dtype)
    assert xp.allclose(p_half, ref_p_half, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="delta", dtype=sp.dtype)
    assert xp.allclose(delta, ref_delta, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="alpha", dtype=sp.dtype)
    assert xp.allclose(alpha, ref_alpha, atol=atol, rtol=rtol)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
@pytest.mark.parametrize(
    "levels,ref_levels",
    [
        (None, {"full": list(range(1, 138)), "half": list(range(0, 138))}),
        (list(range(90, 138)), {"full": list(range(90, 138)), "half": list(range(89, 138))}),
        (list(range(137, 90, -1)), {"full": list(range(137, 90, -1)), "half": list(range(137, 89, -1))}),
        ([1, 2], {"full": [1, 2], "half": [0, 1, 2]}),
        ([2, 1], {"full": [2, 1], "half": [2, 1, 0]}),
    ],
)
@pytest.mark.parametrize(
    "output",
    [
        "full",
        "delta",
        "alpha",
        ["full", "half", "delta", "alpha"],
        ["full", "half"],
        ["half", "full"],
        ["delta", "alpha"],
        ["full", "level"],
        ["delta", "level"],
        ["alpha", "level"],
        ["full", "half", "delta", "alpha", "level"],
        ["full", "half", "level"],
        ["half", "full", "level"],
        ["delta", "alpha", "level"],
    ],
)
def test_pressure_on_hybrid_levels_output_generic(vertical, index, levels, ref_levels, output, xp, device):

    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x, device=device) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    # sp test data is 1D
    sp = sp[index[1]]

    ref_def = {
        "full": ref_p_full[index],
        "half": ref_p_half[index],
        "delta": ref_delta[index],
        "alpha": ref_alpha[index],
    }
    ref = {
        key: val
        for key, val in ref_def.items()
        if (output == key or (isinstance(output, (list, tuple)) and key in output))
    }

    levels = np.asarray(levels) if levels is not None and not isinstance(levels, slice) else levels
    ref_levels = {k: xp.asarray(v, device=device, dtype=int) for k, v in ref_levels.items()}

    for key in ref:
        if key == "half":
            ref[key] = ref[key][ref_levels["half"]]
        else:
            ref[key] = ref[key][ref_levels["full"] - 1]

    res = vertical.pressure_on_hybrid_levels(sp, A, B, levels=levels, alpha_top="ifs", output=output)

    if isinstance(output, str):
        output = [output]
    else:
        output = list(output)

    # If output includes "level", then res is a tuple of (outputs..., levels), where levels
    # is a dict with keys "full" and "half".
    if "level" in output:
        assert isinstance(res, tuple)
        assert len(res) == len(output)
        level_idx = output.index("level")
        res_levels = res[level_idx]

        assert set(res_levels.keys()) == set(ref_levels.keys())
        for k in res_levels:
            assert res_levels[k].shape == ref_levels[k].shape, (
                f"{k=}, expected shape={ref_levels[k].shape}, got={res_levels[k].shape}"
            )
            assert xp.allclose(res_levels[k], ref_levels[k]), f"{k=}, expected={ref_levels[k]}, got={res_levels[k]}"

    if len(output) == 1:
        res = (res,)  # make it a tuple for uniform processing below

    # atol and rtol for different outputs, due to different precisions in backends
    tolerance = Tolerance({
        "full": {64: (1e-8, 1e-6)},
        "half": {64: (1e-8, 1e-6)},
        "delta": {64: (1e-8, 1e-6), 32: (1e-6, 1e-5)},
        "alpha": {64: (1e-8, 1e-6), 32: (1e-4, 1e-5)},
    })

    assert isinstance(res, tuple)
    assert len(res) == len(output)
    for key, rd in zip(output, res):
        if key != "level":
            atol, rtol = tolerance.get(key=key, dtype=sp.dtype)
            assert rd.shape == ref[key].shape, f"{key=}, expected shape={ref[key].shape}, got={rd.shape}"
            assert xp.allclose(rd, ref[key], atol=atol, rtol=rtol), (
                f"{key=}, max abs diff={xp.max(xp.abs(rd - ref[key]))}"
            )


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
@pytest.mark.parametrize(
    "levels,ref_levels",
    [
        (None, {"full": list(range(1, 138)), "half": list(range(0, 138))}),
        (list(range(90, 138)), {"full": list(range(91, 138)), "half": list(range(90, 138))}),
        (list(range(137, 90, -1)), {"full": list(range(137, 91, -1)), "half": list(range(137, 90, -1))}),
        ([1, 2], {"full": [2], "half": [1, 2]}),
        ([2, 1], {"full": [2], "half": [2, 1]}),
        ([0, 1, 2], {"full": [1, 2], "half": [0, 1, 2]}),
        ([2, 1, 0], {"full": [2, 1], "half": [2, 1, 0]}),
    ],
)
def test_pressure_on_hybrid_levels_output_half_only(vertical, index, levels, ref_levels, xp, device):
    """Test pressure_on_hybrid_levels with half_only=True and output='half' or ['half'].
    This is a special case, because only half levels are computed, and the
    levels kwarg in this case refers to half levels, not full levels as in the general case.
    """
    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_half = DATA_HYBRID_CORE.p_half

    sp, ref_p_half, A, B = (xp.asarray(x, device=device) for x in [sp, ref_p_half, A, B])

    # sp test data is 1D
    sp = sp[index[1]]
    ref_p_half = ref_p_half[index]
    levels = np.asarray(levels) if levels is not None and not isinstance(levels, slice) else levels
    ref_levels = {k: xp.asarray(v, device=device, dtype=int) for k, v in ref_levels.items()}
    ref_vals = ref_p_half[ref_levels["half"]]

    res = vertical.pressure_on_hybrid_levels(sp, A, B, levels=levels, alpha_top="ifs", output=["half", "level"])

    assert isinstance(res, tuple)
    assert len(res) == 2
    res_levels = res[-1]
    res_half = res[0]

    assert set(res_levels.keys()) == set(ref_levels.keys())
    for k in res_levels:
        assert res_levels[k].shape == ref_levels[k].shape, (
            f"{k=}, expected shape={ref_levels[k].shape}, got={res_levels[k].shape}"
        )
        assert xp.allclose(res_levels[k], ref_levels[k]), f"{k=}, expected={ref_levels[k]}, got={res_levels[k]}"

    # atol and rtol for different outputs, due to different precisions in backends
    tolerance = Tolerance({
        "half": {64: (1e-8, 1e-6)},
    })

    atol, rtol = tolerance.get(key="half", dtype=sp.dtype)
    assert res_half.shape == ref_vals.shape, f"expected shape={ref_vals.shape}, got={res_half.shape}"
    assert xp.allclose(res_half, ref_vals, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(res_half - ref_vals))}"


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_alpha_delta(vertical, index, xp, device):

    alpha = DATA_HYBRID_CORE.alpha
    delta = DATA_HYBRID_CORE.delta
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, alpha, delta = (xp.asarray(x, device=device) for x in [z_ref, t, q, alpha, delta])

    alpha = alpha[index]
    delta = delta[index]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(t, q, alpha, delta)

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (1e-8, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(z, z_ref, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(z - z_ref))}"


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_ab(vertical, index, xp, device):

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, sp, A, B = (xp.asarray(x, device=device) for x in [z_ref, t, q, sp, A, B])

    sp = sp[index[1]]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, sp, A, B)

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(z, z_ref, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(z - z_ref))}"


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_part(vertical, index, xp, device):
    # get only levels from 90 to 136/137
    part = slice(90, None)

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, sp, A, B = (xp.asarray(x, device=device) for x in [z_ref, t, q, sp, A, B])

    part_index = (part, index[1])

    sp = sp[index[1]]
    t = t[part_index]
    q = q[part_index]
    z_ref = z_ref[part_index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, sp, A, B)

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(z, z_ref, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(z - z_ref))}"


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_geopotential_on_hybrid_levels(vertical, index, xp, device):

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z
    zs = [0.0] * len(t[0])  # surface geopotential is zero in test data

    z_ref, t, q, zs, sp, A, B = (xp.asarray(x, device=device) for x in [z_ref, t, q, zs, sp, A, B])

    sp = sp[index[1]]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]
    zs = zs[index[1]]

    z = vertical.geopotential_on_hybrid_levels(t, q, zs, sp, A, B)

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(z, z_ref, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(z - z_ref))}"


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
@pytest.mark.parametrize("h_type", ["geometric", "geopotential"])
@pytest.mark.parametrize("h_reference", ["sea", "ground"])
def test_height_on_hybrid_levels(vertical, index, xp, device, h_type, h_reference):

    A, B = vertical.hybrid_level_parameters(137)
    A = A.tolist()
    B = B.tolist()

    sp = DATA_HYBRID_H.p_surf
    t = DATA_HYBRID_H.t
    q = DATA_HYBRID_H.q
    zs = DATA_HYBRID_H.z_surf

    ref_name = f"h_{h_type}_{h_reference}"
    h_ref = getattr(DATA_HYBRID_H, ref_name)

    h_ref, t, q, zs, sp, A, B = (xp.asarray(x, device=device) for x in [h_ref, t, q, zs, sp, A, B])

    t = t[index]
    q = q[index]
    h_ref = h_ref[index]
    zs = zs[index[1]]
    sp = sp[index[1]]

    h = vertical.height_on_hybrid_levels(t, q, zs, sp, A, B, h_type=h_type, h_reference=h_reference)

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(h, h_ref, atol=atol, rtol=rtol), f"max abs diff={xp.max(xp.abs(h - h_ref))}"


from earthkit.meteo import constants

# column of 3 levels, top first; a second column is the first shifted by 100 m
_EXTRAP_Z = [[3000.0, 3100.0], [2000.0, 2100.0], [1000.0, 1100.0]]
_EXTRAP_T = [[260.0, 262.0], [270.0, 271.0], [280.0, 284.0]]


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "mode,target_z,expected",
    [
        ("constant", [500.0, 600.0], [280.0, 284.0]),
        ("linear", [500.0, 600.0], [285.0, 290.5]),
        ("lapse_rate", [500.0, 600.0], [283.25, 287.25]),
        ("lapse_rate", [1000.0, 1100.0], [280.0, 284.0]),
        ("lapse_rate", [1500.0, np.nan], [np.nan, np.nan]),
    ],
)
def test_extrapolate_below_lowest_level(vertical, mode, target_z, expected, xp, device):
    data = xp.asarray(_EXTRAP_T, device=device)
    z = xp.asarray(_EXTRAP_Z, device=device)
    target_z = xp.asarray(target_z, device=device)
    expected = xp.asarray(expected, device=device)

    r = vertical.extrapolate_below_lowest_level(data, z, target_z[xp.newaxis], mode=mode)
    assert r.shape == (1, 2)
    assert xp.allclose(r[0], expected, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_extrapolate_below_lowest_level_shapes(vertical, xp, device):
    # levels bottom first, on the last axis, 1D z, scalar and 1D targets
    data = xp.asarray([[280.0, 270.0, 260.0], [284.0, 271.0, 262.0]], device=device)
    z = xp.asarray([1000.0, 2000.0, 3000.0], device=device)

    r = vertical.extrapolate_below_lowest_level(data, z, 0.0, vertical_dim=1)
    assert xp.allclose(r, xp.asarray([286.5, 290.5], device=device))

    r = vertical.extrapolate_below_lowest_level(data, z, [0.0, 1500.0], mode="linear", vertical_dim=1)
    assert r.shape == (2, 2)
    assert xp.allclose(r, xp.asarray([[290.0, np.nan], [297.0, np.nan]], device=device), equal_nan=True)


def test_extrapolate_below_lowest_level_bad_mode(vertical):
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(np.zeros(3), np.arange(3.0), 0.0, mode="hydstat")
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(np.zeros(1), np.zeros(1), 0.0, mode="linear")


_HYD_P0 = [90000.0, 89000.0]
_HYD_T0 = [280.0, 270.0]
_HYD_DZ = -500.0  # targets 500 m below a start level at 1000 m / 1100 m


def _hyd_ct(p0, t0):
    return p0 * np.exp(-constants.g / (constants.Rd * t0) * _HYD_DZ)


def _hyd_clr(p0, t0, gamma):
    return p0 * (1.0 - gamma * _HYD_DZ / t0) ** (constants.g / (constants.Rd * gamma))


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "lapse_rate,expected",
    [
        (0.0, [_hyd_ct(p, t) for p, t in zip(_HYD_P0, _HYD_T0)]),
        (0.0065, [_hyd_clr(p, t, 0.0065) for p, t in zip(_HYD_P0, _HYD_T0)]),
        # an inversion takes the isothermal form
        (-0.01, [_hyd_ct(p, t) for p, t in zip(_HYD_P0, _HYD_T0)]),
    ],
)
def test_extrapolate_below_lowest_level_hydrostatic(vertical, lapse_rate, expected, xp, device):
    data = xp.asarray(_EXTRAP_T, device=device)
    z = xp.asarray(_EXTRAP_Z, device=device)
    start_z = xp.asarray([1000.0, 1100.0], device=device)
    target_z = start_z + _HYD_DZ

    r = vertical.extrapolate_below_lowest_level(
        data,
        z,
        target_z[xp.newaxis],
        mode="hydrostatic",
        lapse_rate=lapse_rate,
        start_z=start_z,
        start_value=xp.asarray(_HYD_P0, device=device),
        t_start=xp.asarray(_HYD_T0, device=device),
    )
    assert r.shape == (1, 2)
    assert xp.allclose(r[0], xp.asarray(expected, device=device), rtol=1e-6)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_extrapolate_below_lowest_level_start_above_lowest(vertical, xp, device):
    # start level 500 m above the lowest model level: it overrides the model values below it
    data = xp.asarray(_EXTRAP_T, device=device)
    z = xp.asarray(_EXTRAP_Z, device=device)
    start_z = xp.asarray([1500.0, 1600.0], device=device)
    start_value = xp.asarray([275.0, 279.0], device=device)
    target_z = xp.asarray([[1200.0, 1700.0], [500.0, 600.0]], device=device)

    r = vertical.extrapolate_below_lowest_level(
        data, z, target_z, mode="constant", start_z=start_z, start_value=start_value
    )
    assert xp.allclose(r, xp.asarray([[275.0, np.nan], [275.0, 279.0]], device=device), equal_nan=True)

    vgrad = xp.asarray([-0.01, 0.002], device=device)
    r = vertical.extrapolate_below_lowest_level(
        data, z, target_z, lapse_rate=-vgrad, start_z=start_z, start_value=start_value
    )
    assert xp.allclose(r, xp.asarray([[278.0, np.nan], [285.0, 277.0]], device=device), equal_nan=True)


def test_extrapolate_below_lowest_level_bad_combinations(vertical):
    data, z = np.zeros(3), np.arange(3.0)
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(data, z, 0.0, start_z=1.0)
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(data, z, 0.0, start_value=1.0)
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(data, z, 0.0, mode="linear", start_z=1.0, start_value=1.0)
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(data, z, 0.0, mode="hydrostatic")
    with pytest.raises(ValueError):
        vertical.extrapolate_below_lowest_level(data, z, 0.0, mode="constant", t_start=280.0)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("mode", ["lapse_rate", "constant", "linear"])
def test_extrapolate_below_lowest_level_dtype(vertical, xp, device, dtype, mode):
    data = xp.asarray([[280.0, 281.0], [275.0, 276.0], [270.0, 271.0]], dtype=dtype, device=device)
    z = xp.asarray([100.0, 600.0, 1100.0], dtype=dtype, device=device)
    r = vertical.extrapolate_below_lowest_level(data, z, xp.asarray([0.0], dtype=dtype, device=device), mode=mode)
    assert r.dtype == data.dtype
    assert r.shape == (1, 2)
