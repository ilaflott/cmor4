"""Public dataset creation, file access, and output naming entry points."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .axis import Axis
from .datasetinfo import DatasetInfo
from .grid import Grid
from . import _assembly, _netcdf
from ._assembly import ValidationContext
from ._metadata import INTERNAL_DATASET_KEYS, DatasetMetadata
from ._templates import render_template
from ._time import add_time_delta, date_part, decode_time_value
from .variable import Variable
from .zfactor import ZFactor

DEFAULT_OUTPUT_PATH_TEMPLATE = (
    "<drs_specs><mip_era><activity_id><institution_id><source_id>"
    "<experiment_id><variant_label><region><frequency><variable_id>"
    "<branding_suffix><grid_label><version>"
)

DEFAULT_OUTPUT_FILE_TEMPLATE = (
    "<branded_variable><frequency><region><grid_label><source_id>"
    "<experiment_id><variant_label><time_range>"
)


def create_dataset(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    path: str | Path | None = None,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Validate metadata and data, then write one variable to a NetCDF file.

    Identical to :func:`cmorize`; see it for full details.

    Parameters
    ----------
    dataset:
        Dataset metadata.
    variable:
        Variable metadata.
    axes:
        Coordinate axes.
    data:
        Data values.
    zfactors:
        Formula terms for a hybrid vertical coordinate.
    grid:
        Grid for curvilinear or projected data.
    path:
        Output file path.
    attrs:
        Extra global attributes.
    encoding:
        NetCDF encoding options.
    **to_netcdf_kwargs:
        Passed to :meth:`xarray.Dataset.to_netcdf`.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """

    ctx = _assembly.validate_metadata(dataset, variable, axes, zfactors, grid)
    data_array = _assembly.validate_data_chunk(ctx, data)
    ds = create_dataset_from_validated_data(
        ctx,
        data_array,
        attrs=attrs,
        encoding=encoding,
    )
    return write_netcdf(ds, ctx.dataset, ctx.variable, path=path, **to_netcdf_kwargs)


def create_dataset_from_validated_data(
    ctx: ValidationContext,
    data_array: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create a dataset from metadata and data that was already validated.

    This internal helper keeps final DatasetWriter construction on the same
    path as :func:`create_dataset` without forcing a lazy staged array into
    memory for a second validation pass.
    """

    return _assembly.build_dataset(ctx, data_array, attrs=attrs, encoding=encoding)


def write_netcdf(
    ds: xr.Dataset,
    dataset: DatasetMetadata,
    variable: Variable,
    path: str | Path | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Write an xarray dataset to NetCDF.

    Most users call :func:`cmorize` instead, which builds the dataset and
    then calls this function.

    Parameters
    ----------
    ds:
        Dataset to write.
    dataset:
        Dataset metadata. Used to build the default path and to read
        ``create_subdirectories``.
    variable:
        Variable metadata. Used to build the default path.
    path:
        Output file path. If omitted, CMOR builds it with
        :func:`build_output_path`.
    **to_netcdf_kwargs:
        Passed to :meth:`xarray.Dataset.to_netcdf`.

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    ValueError
        If ``create_subdirectories`` is ``False`` and the output directory
        does not exist. By default CMOR creates missing directories.
    """

    output_path = (
        Path(path) if path is not None else build_output_path(dataset, variable, ds)
    )
    return _netcdf.write_dataset(ds, dataset, output_path, **to_netcdf_kwargs)


def cmorize(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    path: str | Path | None = None,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Validate metadata and data, then write one variable to a NetCDF file.

    When the inputs come from :class:`ProjectTables`, CMOR checks them
    against the project tables and controlled vocabulary, fills default
    global attributes, and adds any scalar coordinates the table requires.
    Objects created directly skip the table and CV checks.

    See :doc:`/usage/basic_usage` for a complete example.

    Parameters
    ----------
    dataset:
        Dataset metadata from :meth:`ProjectTables.dataset_info`.
    variable:
        Metadata for the variable to write, usually from
        :meth:`ProjectTables.variable`.
    axes:
        One :class:`Axis` per variable dimension, plus any auxiliary or
        scalar coordinates.
    data:
        Array of data values. Its shape must match the variable dimensions.
    zfactors:
        Formula terms for a hybrid vertical coordinate. See
        :doc:`/usage/vertical_coordinates`.
    grid:
        Grid for curvilinear or projected data. See :doc:`/usage/grids`.
    path:
        Output file path. If omitted, CMOR builds it with
        :func:`build_output_path`.
    attrs:
        Extra global attributes to add to the file.
    encoding:
        NetCDF encoding options, keyed by output variable name. For CMIP7,
        CMOR sets chunk sizes automatically; chunk sizes you supply are
        checked against the CMIP7 rules.
    **to_netcdf_kwargs:
        Passed to :meth:`xarray.Dataset.to_netcdf`.

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    TableValidationError
        If the metadata does not match the project tables.
    AxisValidationError
        If coordinate values or bounds are invalid.
    VariableValidationError
        If data values fail the variable's range or missing-value checks.
    ControlledVocabularyError
        If a global attribute is missing or not allowed by the CV.
    ValueError
        If the data shape does not match the dimensions, or chunk sizes do
        not meet the CMIP7 rules.
    """

    return create_dataset(
        dataset,
        variable,
        axes,
        data,
        zfactors=zfactors,
        grid=grid,
        path=path,
        attrs=attrs,
        encoding=encoding,
        **to_netcdf_kwargs,
    )


def open_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF file with :func:`xarray.open_dataset`.

    Parameters
    ----------
    path:
        Path to the NetCDF file.
    **kwargs:
        Passed to :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        The opened dataset.
    """

    return xr.open_dataset(path, **kwargs)


def build_output_path(
    dataset: DatasetMetadata,
    variable: Variable,
    ds: xr.Dataset | None = None,
) -> Path:
    """Build the output file path from dataset and variable metadata.

    The directory and file name come from templates. CMOR uses the first of
    these that is set:

    1. ``output_path_template`` and ``output_file_template`` in the dataset
       metadata.
    2. ``directory_path_template`` and ``filename_template`` in the
       ``DRS`` section of the CV. See :ref:`cv-special-sections`.
    3. CMOR's built-in default templates.

    The path is relative to the dataset's ``outpath`` (default: the current
    directory).

    Parameters
    ----------
    dataset:
        Dataset metadata.
    variable:
        Variable metadata.
    ds:
        Dataset to read the time range from, for the ``<time_range>``
        token. If omitted, the file name has no time range.

    Returns
    -------
    pathlib.Path
        Full path to the output ``.nc`` file.
    """

    if ds is not None and ds.attrs.get("frequency"):
        dataset = dataset.model_copy(update={"frequency": ds.attrs["frequency"]})
    dataset, variable = _assembly.dataset_for_variable(dataset, variable)
    root = Path(str(dataset.outpath or ".")).expanduser()
    tokens = _template_tokens(dataset, variable, ds)

    # Read CV DRS templates once; both may be None when the CV lacks a DRS
    # section (e.g. obs4MIPs) or when the CV is not project-backed.
    cv_path_tmpl: str | None = None
    cv_file_tmpl: str | None = None
    cv = getattr(getattr(dataset, "project", None), "cv", None)
    if cv is not None and cv._drs is not None:
        cv_path_tmpl, cv_file_tmpl = cv._drs.templates()

    path_template = str(
        dataset.output_path_template or cv_path_tmpl or DEFAULT_OUTPUT_PATH_TEMPLATE
    )
    file_template = str(
        dataset.output_file_template or cv_file_tmpl or DEFAULT_OUTPUT_FILE_TEMPLATE
    )

    if (
        tokens.get("time_range")
        and "<time_range>" not in file_template
        and "<time-range>" not in file_template
    ):
        file_template += "<time_range>"

    directory = render_template(path_template, tokens, "/").strip("/")
    filename = render_template(file_template, tokens, "_") + ".nc"

    return root / directory / filename


def _template_tokens(
    dataset: DatasetMetadata,
    variable: Variable,
    ds: xr.Dataset | None,
) -> dict[str, Any]:
    var_name, labels = variable.names()
    dataset_values = dataset.to_dict()
    frequency = str(
        (ds.attrs.get("frequency") if ds is not None else None)
        or dataset.frequency
        or variable.frequency
        or "fx"
    )
    variant_label = dataset.variant_label()
    version = str(dataset.version or f"v{date.today():%Y%m%d}")
    time_range = _time_range(ds, frequency) if frequency != "fx" else None

    tokens = {
        str(key): value
        for key, value in (ds.attrs.items() if ds is not None else ())
        if not str(key).startswith("_")
    }
    tokens.update({
        str(key): value
        for key, value in dataset_values.items()
        if (key not in INTERNAL_DATASET_KEYS and not str(key).startswith("_"))
    })
    tokens.update({
        "branded_name": labels["branded_name"],
        "branded_variable": labels["branded_name"],
        "branded_variable_name": labels["branded_name"],
        "branding_suffix": labels.get("branding_suffix", ""),
        "frequency": frequency,
        "grid_label": dataset.grid_label or tokens.get("grid_label", "gn"),
        "member_id": dataset.member_id or variant_label,
        "region": dataset.region or tokens.get("region", "glb"),
        "time-range": time_range or "",
        "time_range": time_range or "",
        "variable_id": var_name,
        "variant_label": variant_label,
        "version": version,
    })
    for key in (
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
    ):
        if key in labels:
            tokens[key] = labels[key]
    return tokens


def _time_range(ds: xr.Dataset | None, frequency: str = "mon") -> str | None:
    if ds is None or "time" not in ds.coords:
        return None
    time = ds["time"]
    units = time.attrs.get("units")
    calendar = time.attrs.get("calendar", "standard")
    climatology_bounds_name = time.attrs.get("climatology")
    climatology = bool(climatology_bounds_name)
    if climatology:
        if str(climatology_bounds_name) not in ds:
            return None
        bounds = np.asarray(ds[str(climatology_bounds_name)].values)
        if bounds.size == 0:
            return None
        bounds = bounds.reshape(-1, bounds.shape[-1])
        first_value = bounds[0, 0]
        last_value = bounds[-1, -1]
    else:
        values = np.asarray(time.values)
        if values.size == 0:
            return None
        first_value = values.flat[0]
        last_value = values.flat[-1]
    first = decode_time_value(first_value, units, calendar)
    last = decode_time_value(last_value, units, calendar)
    if first is None or last is None:
        return None
    if climatology:
        first = add_time_delta(first, timedelta(hours=1))
        last = add_time_delta(last, timedelta(hours=-1))
    freq = frequency.lower()
    clim_suffix = (
        "-clim"
        if climatology and str(ds.attrs.get("mip_era", "")).upper() != "CMIP7"
        else ""
    )
    if "yr" in freq or "dec" in freq:
        return f"{date_part(first, 'year')}-{date_part(last, 'year')}{clim_suffix}"
    if "monc" in freq or "mon" in freq or climatology:
        return f"{date_part(first, 'month')}-{date_part(last, 'month')}{clim_suffix}"
    if "day" in freq:
        return f"{date_part(first, 'day')}-{date_part(last, 'day')}{clim_suffix}"
    if "subhr" in freq:
        return f"{date_part(first, 'second')}-{date_part(last, 'second')}{clim_suffix}"
    if "hr" in freq or freq in {"hour", "hourly"}:
        return f"{date_part(first, 'minute')}-{date_part(last, 'minute')}{clim_suffix}"
    return f"{date_part(first, 'month')}-{date_part(last, 'month')}{clim_suffix}"
