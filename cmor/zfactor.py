"""ZFactor metadata record for hybrid-coordinate formula terms."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from ._metadata import CoercedF, MetadataModel, StrSeq


class ZFactor(MetadataModel):
    """Values and metadata for one formula term of a hybrid vertical coordinate.

    Usually built with :meth:`ProjectTables.zfactor`, which fills these
    fields from the formula-terms table::

        ps = project.zfactor("ps", values=surface_pressure)

    See :doc:`/usage/vertical_coordinates` for a full example.

    Parameters
    ----------
    name:
        Formula-term name, such as ``"a"``, ``"b"``, or ``"ps"``.
    values:
        Values of the term: a scalar or an array.
    dimensions:
        Dimension names, in the order of ``values``.
    units:
        Units of the term's values, such as ``"Pa"``. A value you give must
        match the table.
    standard_name:
        CF standard name, such as ``"surface_air_pressure"``.
    long_name:
        Human-readable description of the term.
    out_name:
        Name of the term's variable in the output file.
    table_entry:
        Formula-table entry to use, if different from ``name``.
    bounds:
        Bounds of the term, for terms that apply to level bounds.
    bounds_name:
        Name of the bounds variable in the output file. Defaults to
        ``"<out_name>_bnds"``.
    bounds_dim:
        Name of the bounds variable's second dimension (the one that holds
        each cell's lower and upper bound). Defaults to ``"bnds"``.
    bounds_attrs:
        Extra attributes for the bounds variable.
    valid_min:
        Smallest expected value. CMOR warns about values below it.
    valid_max:
        Largest expected value. CMOR warns about values above it.
    ok_min_mean_abs:
        Smallest expected mean of the absolute values, checked as for
        :class:`Variable`.
    ok_max_mean_abs:
        Largest expected mean of the absolute values, checked as for
        :class:`Variable`.
    attrs:
        Extra attributes for the term's variable.
    """

    name: str
    values: Any = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    out_name: str | None = None
    table_entry: str | None = None
    bounds: Any = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: dict[str, Any] = Field(default_factory=dict)
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    ok_min_mean_abs: CoercedF = None
    ok_max_mean_abs: CoercedF = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term variable."""
        attrs = self.netcdf_attrs(self.attrs)
        for key, val in (
            ("units", self.units),
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
        ):
            if val is not None:
                attrs[key] = val
        return attrs

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return formula-term values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.values if self.values is not None else [])

    def bounds_array(self) -> np.ndarray:
        """Return formula-term bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.bounds)
