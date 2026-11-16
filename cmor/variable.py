"""Variable metadata, data and formula-term limits, and value diagnostics."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Sequence, Union

import numpy as np
from pydantic import BeforeValidator, Field

from ._metadata import CoercedF, IntTuple, MetadataModel, StrOrTuple, StrSeq
from ._metadata import numeric_or_none as _numeric_or_none
from .exceptions import VariableValidationError

if TYPE_CHECKING:
    from .axis import Axis
    from .zfactor import ZFactor


# Literal type for the CF positive attribute, accepting any case.
PositiveLiteral = Annotated[
    Literal["up", "down"] | None,
    BeforeValidator(lambda value: value.lower() if isinstance(value, str) else value),
]

# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class Variable(MetadataModel):
    """Metadata for the variable being written.

    Usually built with :meth:`ProjectTables.variable`, which fills these
    fields from the variable table::

        variable = project.variable("tas")

    You can also build one directly if you are not using project tables::

        variable = Variable(name="tas", units="K", dimensions=("time", "lat", "lon"))

    Parameters
    ----------
    name:
        Variable name or branded variable name.
    id:
        Name of the variable in the output file. Defaults to the table's
        ``out_name``, or the part of ``name`` before the first ``_``. A value
        you give must match the table.
    variable_id:
        Alternative to ``id``; ``id`` is used if both are given. Also
        written as the ``variable_id`` global attribute.
    table_id:
        Table to use when the variable is in more than one table.
    dimensions:
        Dimension names, in the order of the data array.
    units:
        Units of the data.
    standard_name:
        CF standard name, such as ``"air_temperature"``.
    long_name:
        Human-readable description of the variable.
    cell_methods:
        CF ``cell_methods``: how values were computed over each cell, such
        as ``"area: time: mean"``.
    cell_measures:
        CF ``cell_measures``: the variable that holds cell areas or volumes,
        such as ``"area: areacella"``.
    comment:
        Free-text notes about the variable.
    flag_values:
        Values used in flag (category) data, such as ``[0, 1, 2]``.
    flag_meanings:
        Space-separated meaning of each flag value, in the same order as
        ``flag_values``.
    missing_value:
        Value that marks missing data. CMOR skips these points when it checks
        the data, and writes the value as ``_FillValue`` and
        ``missing_value``.
    fill_value:
        Alternative to ``missing_value``; ``missing_value`` is used if both
        are given.
    chunksizes:
        NetCDF chunk size for each dimension. Chunk sizes in ``encoding``
        take priority.
    chunks:
        Alternative to ``chunksizes``; ``chunksizes`` is used if both are
        given.
    coordinates:
        Value for the ``coordinates`` attribute. CMOR normally sets it.
    formula_terms:
        Value for the ``formula_terms`` attribute. CMOR normally sets it.
    positive:
        ``"up"`` or ``"down"``, for vertical fluxes.
    frequency:
        Sampling frequency, such as ``"mon"``. Written as the ``frequency``
        global attribute.
    realm:
        Model component, such as ``"atmos"`` or ``"ocean"``. Written as the
        ``realm`` global attribute.
    table_info:
        Description of the table file the variable came from. Written as the
        ``table_info`` global attribute.
    valid_min:
        Smallest expected data value. CMOR warns about values below it.
    valid_max:
        Largest expected data value. CMOR warns about values above it.
    ok_min_mean_abs:
        Smallest expected mean of the absolute data values. CMOR warns if
        the mean is below it, and raises an error if the mean is less than
        a tenth of it.
    ok_max_mean_abs:
        Largest expected mean of the absolute data values. CMOR warns if the
        mean is above it, and raises an error if the mean is more than 10
        times it.
    attrs:
        Extra attributes to write on the variable.
    """

    name: str
    id: str | None = None
    variable_id: str | None = None
    table_id: str | None = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    flag_values: str | None = None
    flag_meanings: str | None = None
    missing_value: float | int | None = None
    fill_value: float | int | None = None
    chunksizes: IntTuple = None
    chunks: IntTuple = None
    coordinates: StrOrTuple = None
    formula_terms: str | None = None
    positive: PositiveLiteral = None
    frequency: str | None = None
    realm: str | None = None
    table_info: str | None = None
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    ok_min_mean_abs: CoercedF = None
    ok_max_mean_abs: CoercedF = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def names(self) -> tuple[str, dict[str, str]]:
        """Return ``(variable_id, labels_dict)``.

        ``labels_dict`` always contains ``"branded_name"`` and
        ``"variable_id"``; optional keys are ``"branding_suffix"``,
        ``"temporal_label"``, ``"vertical_label"``, ``"horizontal_label"``,
        and ``"area_label"``.
        """
        branded = self.name
        var_id = str(self.id or self.variable_id or branded.split("_", 1)[0])
        labels: dict[str, str] = {"branded_name": branded, "variable_id": var_id}
        if "_" in branded:
            suffix = branded.split("_", 1)[1]
            labels["branding_suffix"] = suffix
            for key, val in zip(
                ("temporal_label", "vertical_label", "horizontal_label", "area_label"),
                suffix.split("-"),
            ):
                labels[key] = val
        return var_id, labels

    def attributes(self, labels: Mapping[str, str]) -> dict[str, Any]:
        """Return NetCDF attributes for this data variable."""
        attrs = self.netcdf_attrs(self.attrs)
        for key, val in (
            ("units", self.units),
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
            ("cell_methods", self.cell_methods),
            ("cell_measures", self.cell_measures),
            ("comment", self.comment),
            ("positive", self.positive),
            ("flag_values", self.flag_values),
            ("flag_meanings", self.flag_meanings),
        ):
            if val is not None:
                attrs[key] = val
        attrs.setdefault("branded_variable_name", labels["branded_name"])
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                attrs.setdefault(key, labels[key])
        return attrs


def validate_variable_values(
    variable: Union[Variable, ZFactor],
    axes: Sequence[Axis],
    data: Any,
    dims: Sequence[str],
    axis_dims: Mapping[str, tuple[str, ...]],
    *,
    name: str | None = None,
    table_id: str | None = None,
) -> None:
    """Apply validation to data variable and formula-term values."""

    values: np.ma.MaskedArray
    try:
        values = np.ma.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return

    valid_mask = ~np.ma.getmaskarray(values)
    missing_value = getattr(variable, "missing_value", None) or getattr(
        variable, "fill_value", None
    )
    if missing_value is not None:
        try:
            valid_mask &= ~np.isclose(
                values.filled(np.nan),
                float(missing_value),
                rtol=float(variable.extra.get("tolerance", 1.0e-6)),
                atol=0.0,
                equal_nan=False,
            )
        except (TypeError, ValueError):
            pass

    numeric = values.filled(np.nan)
    nan_mask = np.isnan(numeric) & valid_mask
    if np.any(nan_mask):
        count = int(np.count_nonzero(nan_mask))
        index = _first_index(nan_mask)
        raise VariableValidationError(
            "Invalid value(s) detected for variable "
            f"{_variable_name(variable, name)!r} "
            f"(table: {_table_id(variable, table_id)}): "
            f"{count} values were NaNs. First encountered NaN was at "
            "(axis: index/value):"
            f"{_format_location(index, dims, axes, axis_dims)}"
        )

    active = numeric[valid_mask]
    active = active[np.isfinite(active)]
    if active.size == 0:
        return

    _warn_for_limit(
        variable,
        numeric,
        valid_mask,
        dims,
        axes,
        axis_dims,
        "valid_min",
        np.less,
        "lower than minimum valid value",
        np.nanmin,
        name=name,
        table_id=table_id,
    )
    _warn_for_limit(
        variable,
        numeric,
        valid_mask,
        dims,
        axes,
        axis_dims,
        "valid_max",
        np.greater,
        "greater than maximum valid value",
        np.nanmax,
        name=name,
        table_id=table_id,
    )
    _check_absolute_mean(variable, active, name=name, table_id=table_id)


def _warn_for_limit(
    variable: Union[Variable, ZFactor],
    numeric: np.ndarray,
    valid_mask: np.ndarray,
    dims: Sequence[str],
    axes: Sequence[Axis],
    axis_dims: Mapping[str, tuple[str, ...]],
    key: str,
    compare: Any,
    phrase: str,
    extrema: Any,
    *,
    name: str | None,
    table_id: str | None,
) -> None:
    limit = _numeric_or_none(getattr(variable, key, None))
    if limit is None:
        return
    bad_mask = compare(numeric, limit) & valid_mask
    if not np.any(bad_mask):
        return
    count = int(np.count_nonzero(bad_mask))
    bad_values = np.where(bad_mask, numeric, np.nan)
    bad_value = float(extrema(bad_values))
    index = _first_index(bad_mask)
    warnings.warn(
        "Invalid value(s) detected for variable "
        f"{_variable_name(variable, name)!r} "
        f"(table: {_table_id(variable, table_id)}): "
        f"{count} values were {phrase} ({limit:.4g}). "
        f"Encountered bad value ({bad_value:.5g}) was at "
        "(axis: index/value):"
        f"{_format_location(index, dims, axes, axis_dims)}",
        RuntimeWarning,
        stacklevel=3,
    )


def _check_absolute_mean(
    variable: Union[Variable, ZFactor],
    active: np.ndarray,
    *,
    name: str | None,
    table_id: str | None,
) -> None:
    mean_abs = float(np.mean(np.abs(active)))
    ok_min = _numeric_or_none(variable.ok_min_mean_abs)
    if ok_min is not None:
        if mean_abs < 0.1 * ok_min:
            raise VariableValidationError(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is lower by more than an order of "
                f"magnitude than minimum allowed: {ok_min:.4g}"
            )
        if mean_abs < ok_min:
            warnings.warn(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is lower "
                f"than minimum allowed: {ok_min:.4g}",
                RuntimeWarning,
                stacklevel=3,
            )

    ok_max = _numeric_or_none(variable.ok_max_mean_abs)
    if ok_max is not None:
        if mean_abs > 10.0 * ok_max:
            raise VariableValidationError(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is greater by more than an order of "
                f"magnitude than maximum allowed: {ok_max:.4g}"
            )
        if mean_abs > ok_max:
            warnings.warn(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is greater "
                f"than maximum allowed: {ok_max:.4g}",
                RuntimeWarning,
                stacklevel=3,
            )


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.argwhere(mask)[0])


def _format_location(
    index: tuple[int, ...],
    dims: Sequence[str],
    axes: Sequence[Axis],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    axis_by_dim: dict[str, Axis] = {}
    for candidate in axes:
        axis_dimensions = axis_dims.get(candidate.name, ())
        if len(axis_dimensions) == 1:
            axis_by_dim.setdefault(axis_dimensions[0], candidate)
    parts: list[str] = []
    for dim, location in zip(dims, index):
        axis = axis_by_dim.get(str(dim))
        value: Any = location
        if axis is not None:
            values = axis.values_array()
            if values.ndim == 1 and location < values.shape[0]:
                value = values[location]
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, float):
                    value = f"{value:.5g}"
        parts.append(f" {dim}: {location}/{value}")
    return "".join(parts)


def _variable_name(variable: Any, name: str | None) -> str:
    if name is not None:
        return name
    names = getattr(variable, "names", None)
    if callable(names):
        return names()[0]
    return str(getattr(variable, "id", None) or getattr(variable, "name", ""))


def _table_id(variable: Any, table_id: str | None) -> str:
    if table_id is not None:
        return str(table_id)
    return str(getattr(variable, "table_id", "") or "")
