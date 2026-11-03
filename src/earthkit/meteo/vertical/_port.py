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

from .array._port import ceiling_above_ground

__all__ = ["ceiling_above_ground"]
