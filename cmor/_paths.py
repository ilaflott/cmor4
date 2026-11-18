"""Output path resolution: DRS templates, file names, and directories."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import _assembly
from ._metadata import INTERNAL_DATASET_KEYS, DatasetMetadata
from ._templates import render_template
from ._time import add_time_delta, date_part, decode_time_value
from .variable import Variable

DEFAULT_OUTPUT_PATH_TEMPLATE = (
    "<drs_specs><mip_era><activity_id><institution_id><source_id>"
    "<experiment_id><variant_label><region><frequency><variable_id>"
    "<branding_suffix><grid_label><version>"
)

DEFAULT_OUTPUT_FILE_TEMPLATE = (
    "<branded_variable><frequency><region><grid_label><source_id>"
    "<experiment_id><variant_label><time_range>"
)


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


def ensure_output_directory(dataset: DatasetMetadata, output_path: Path) -> None:
    """Create or require the output directory per ``create_subdirectories``."""

    # Honour create_subdirectories.  The flag mirrors CMOR3's own behaviour:
    # when False, the output directory must already exist; CMOR3 errors when
    # it cannot create the directory (e.g. a non-existent /CMIP6 root).
    # Default is True for backwards-compatibility with CMOR's original
    # always-create behaviour.
    create_subdirs = (
        True if dataset.create_subdirectories is None else dataset.create_subdirectories
    )
    if create_subdirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif not output_path.parent.exists():
        raise ValueError(
            f"Output directory {str(output_path.parent)!r} does not exist "
            "and create_subdirectories is disabled. "
            "Create the directory first or set create_subdirectories=True "
            "in the dataset metadata."
        )
