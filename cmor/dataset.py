"""Public entry point for CMORizing a variable to a NetCDF file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .axis import Axis
from .datasetinfo import DatasetInfo
from .grid import Grid
from . import _assembly, _paths
from .variable import Variable
from .zfactor import ZFactor


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
        Output file path. If omitted, CMOR builds the path under the
        dataset's ``outpath`` (default: the current directory) from the first
        templates that are set: ``output_path_template`` and
        ``output_file_template`` in the dataset metadata, then
        ``directory_path_template`` and ``filename_template`` in the ``DRS``
        section of the CV (see :ref:`cv-special-sections`), then CMOR's
        built-in defaults. Missing directories are created unless
        ``create_subdirectories`` is ``False``.
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
        If the data shape does not match the dimensions, chunk sizes do not
        meet the CMIP7 rules, or the output directory does not exist and
        ``create_subdirectories`` is ``False``.
    """

    ctx = _assembly.validate_metadata(dataset, variable, axes, zfactors, grid)
    data_array = _assembly.validate_data_chunk(ctx, data)
    ds = _assembly.build_dataset(ctx, data_array, attrs=attrs, encoding=encoding)
    output_path = (
        Path(path)
        if path is not None
        else _paths.build_output_path(ctx.dataset, ctx.variable, ds)
    )
    _paths.ensure_output_directory(ctx.dataset, output_path)
    ds.to_netcdf(output_path, **to_netcdf_kwargs)
    return output_path
