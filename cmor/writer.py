"""Incremental writer state and its private temporary Zarr storage."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import dask.array as da
import numpy as np
import zarr

from . import _assembly, _netcdf
from ._time import bounds_as_pairs, find_time_axis, metadata_time_axis
from .axis import Axis
from .dataset import build_output_path
from .datasetinfo import DatasetInfo
from .grid import Grid
from .variable import Variable, validate_variable_values
from .zfactor import ZFactor


class DatasetWriter:
    """Write one variable to NetCDF a chunk of time steps at a time.

    Call :meth:`write` once per chunk, then :meth:`close` to finish the file.
    Use the writer as a context manager to close it automatically. Data is
    staged on disk until :meth:`close`, so the full variable never has to fit
    in memory.

    See :doc:`/usage/incremental_writing` for examples.

    Parameters
    ----------
    dataset : DatasetInfo
        Dataset metadata.
    variable : Variable
        Variable metadata.
    axes : Sequence[Axis]
        Coordinate axes. The time axis may have no values; you then pass
        the time values to :meth:`write`.
    path : str or Path, optional
        Output file path. If omitted, CMOR builds it with
        :func:`build_output_path`.
    zfactors : Sequence[ZFactor], optional
        Formula terms for a hybrid vertical coordinate. A term that changes
        with time can have no values; you then pass them to :meth:`write`.
    grid : Grid, optional
        Grid for curvilinear or projected data.
    existing : {"error", "replace", "append"}, default "error"
        What to do if the output file already exists: raise an error,
        overwrite it, or add the new time steps to the end of it.
    encoding : Mapping[str, Any], optional
        NetCDF encoding options, as in :func:`cmorize`.
    attrs : Mapping[str, Any], optional
        Extra global attributes.
    staging_dir : str or Path, optional
        Directory for temporary staging data. Defaults to the system
        temporary directory.
    **to_netcdf_kwargs
        Passed to :meth:`xarray.Dataset.to_netcdf`.

    Attributes
    ----------
    staging_root : Path
        Temporary directory that holds the staged data.
    staging_path : Path
        Location of the staged data. CMOR keeps it after an error so you
        can inspect it.
    """

    def __init__(
        self,
        dataset: DatasetInfo,
        variable: Variable,
        axes: Sequence[Axis],
        path: str | Path | None = None,
        *,
        zfactors: Sequence[ZFactor] | None = None,
        grid: Grid | None = None,
        existing: Literal["error", "replace", "append"] = "error",
        encoding: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        staging_dir: str | Path | None = None,
        **to_netcdf_kwargs: Any,
    ) -> None:
        if existing not in {"error", "replace", "append"}:
            raise ValueError("existing must be one of 'error', 'replace', or 'append'.")

        self.path = Path(path) if path is not None else None
        self.existing = existing
        self.encoding = dict(encoding or {})
        self.attrs = dict(attrs or {})
        self.to_netcdf_kwargs = dict(to_netcdf_kwargs)
        self._closed = False
        self._write_count = 0
        self._time_offset = 0
        self._time_values: list[np.ndarray] = []
        self._time_bounds: list[np.ndarray] = []
        self._saw_time_bounds = False
        self._first_write_after_preserve = False
        self._last_closed_time_value: Any | None = None

        input_axes = tuple(axes)
        input_time_index, input_time_axis = find_time_axis(input_axes)
        metadata_axes = list(input_axes)
        metadata_axes[input_time_index] = metadata_time_axis(input_time_axis)

        self._ctx = _assembly.validate_metadata(
            dataset,
            variable,
            metadata_axes,
            zfactors,
            grid,
        )
        self._time_axis_index, self._time_axis = find_time_axis(
            self._ctx.axes,
        )
        time_dims = self._ctx.axis_dims.get(self._time_axis.name, ())
        if len(time_dims) != 1 or time_dims[0] not in self._ctx.dims:
            raise ValueError(
                "DatasetWriter requires the time axis to map to exactly one "
                "data variable dimension."
            )
        self._time_dim = time_dims[0]
        self._time_dim_index = self._ctx.dims.index(self._time_dim)
        self._initial_time_values = input_time_axis.values_array()
        self._initial_time_bounds = (
            bounds_as_pairs(
                input_time_axis.bounds_array(),
                len(self._initial_time_values),
            )
            if input_time_axis.bounds is not None
            else None
        )

        self._staging = _ZarrStagingStore.create(staging_dir)
        self.staging_root = self._staging.root
        self.staging_path = self._staging.path

    def write(
        self,
        data: Any,
        *,
        time_values: Any | None = None,
        time_bounds: Any | None = None,
        zfactors: Mapping[str, Any] | None = None,
    ) -> None:
        """Add a chunk of time steps to the file.

        Each chunk's times must come after the previous chunk's. If you pass
        bounds, pass them in every write, and each chunk's first bound must
        equal the previous chunk's last bound.

        Parameters
        ----------
        data : array-like
            Data for this chunk. The time dimension may have any length; the
            other dimensions must match the variable.
        time_values : array-like, optional
            Time values for this chunk. Omit them if the time axis already
            has values; the writer then uses the next values from it.
        time_bounds : array-like, optional
            Time bounds for this chunk, in either of these forms:

            - shape ``(N, 2)``: one ``[start, end]`` pair per time step.
            - shape ``(N+1,)``: cell edges.

            Omit them to use bounds from the time axis, if it has any.
        zfactors : Mapping[str, array-like], optional
            Values for time-varying formula terms in this chunk, keyed by
            term name.

        Raises
        ------
        ValueError
            If the writer is closed; the data shape does not match; the times
            do not increase or do not match the data; the bounds are missing,
            the wrong shape, or have gaps; or a formula term is missing,
            unknown, or the wrong shape.
        AxisValidationError
            If the time values are invalid.
        VariableValidationError
            If the data fails the variable's range or missing-value checks.
        """

        self._ensure_open()
        data_array = np.asarray(data)
        chunk_time_values, chunk_time_bounds = self._time_chunk(
            data_array,
            time_values=time_values,
            time_bounds=time_bounds,
        )
        if self._write_count > 0 and self._saw_time_bounds != (
            chunk_time_bounds is not None
        ):
            raise ValueError(
                "time_bounds must be supplied for every write or omitted "
                "for every write."
            )
        chunk_axes = self._axes_with_time(chunk_time_values, chunk_time_bounds)
        chunk_ctx = _assembly.validate_metadata(
            self._ctx.dataset,
            self._ctx.variable,
            chunk_axes,
            self._ctx.zfactors,
            self._ctx.grid,
        )
        validated_data = _assembly.validate_data_chunk(chunk_ctx, data_array)
        self._validate_chunk_shape(validated_data, len(chunk_time_values))
        zfactor_chunks = self._validate_zfactor_chunks(
            zfactors,
            chunk_ctx,
            len(chunk_time_values),
        )
        self._validate_time_order(chunk_time_values, chunk_time_bounds)
        self._staging.append(
            self._ctx.variable.names()[0], validated_data, self._time_dim_index
        )
        self._append_zfactors_to_zarr(zfactor_chunks)

        self._time_values.append(chunk_time_values)
        if chunk_time_bounds is not None:
            self._saw_time_bounds = True
            self._time_bounds.append(chunk_time_bounds)
        self._time_offset += len(chunk_time_values)
        self._write_count += 1
        self._first_write_after_preserve = False

    def close(
        self,
        *,
        preserve_definition: bool = False,
    ) -> Path:
        """Finish and write the NetCDF file.

        Data is copied from the staging area to the file in chunks, so the
        full variable is never loaded into memory. The staging area is
        deleted when this succeeds.

        Parameters
        ----------
        preserve_definition : bool, default False
            If ``True``, keep the writer open after the file is written so
            you can write another file with the same variable and axes. Set
            ``writer.path`` to a new path before the next :meth:`write`. See "Write
            separate segments" in :doc:`/usage/incremental_writing`.

        Returns
        -------
        pathlib.Path
            Absolute path to the written file.

        Raises
        ------
        ValueError
            If the writer is already closed, nothing was written, or bounds
            were given in some writes but not others.
        FileExistsError
            If the file exists and ``existing="error"``.
        AxisValidationError
            If the complete time axis is invalid.
        """

        self._ensure_open()
        var_name = self._ctx.variable.names()[0]
        if self._staging.array(var_name) is None or self._write_count == 0:
            raise ValueError("Cannot close DatasetWriter before any data is written.")

        final_axes = self._final_axes()
        final_ctx = _assembly.validate_metadata(
            self._ctx.dataset,
            self._ctx.variable,
            final_axes,
            self._ctx.zfactors,
            self._ctx.grid,
        )
        final_ctx = replace(final_ctx, zfactors=self._final_zfactors(final_ctx))
        data = self._staging.lazy_array(var_name)
        ds = _assembly.build_dataset(
            final_ctx,
            data,
            attrs=self.attrs,
            encoding=self.encoding,
        )
        output_path = self.path or build_output_path(
            self._ctx.dataset,
            self._ctx.variable,
            ds,
        )
        if self.existing == "append":
            output_path = _netcdf.append_to_existing(
                output_path,
                ds,
                dataset=self._ctx.dataset,
                variable=self._ctx.variable,
                time_dim=self._time_dim,
                time_axis=self._time_axis,
                to_netcdf_kwargs=self.to_netcdf_kwargs,
            )
        elif self.existing == "error" and output_path.exists():
            raise FileExistsError(
                f"Output file {str(output_path)!r} already exists. "
                "Use existing='replace' to overwrite it."
            )
        else:
            output_path = _netcdf.write_dataset(
                ds,
                self._ctx.dataset,
                output_path,
                **self.to_netcdf_kwargs,
            )
        if preserve_definition:
            self._prepare_next_segment()
        else:
            self._closed = True
            self._staging.cleanup()
        return output_path

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> Literal[False]:
        if exc_type is None and not self._closed:
            if self._write_count > 0:
                self.close()
            elif self._first_write_after_preserve:
                self._closed = True
                self._staging.cleanup()
            else:
                self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("DatasetWriter is already closed.")

    @property
    def _zarr_array(self) -> Any | None:
        return self._staging.array(self._ctx.variable.names()[0])

    def _prepare_next_segment(self) -> None:
        if self._time_values and self._time_values[-1].size:
            value = self._time_values[-1][-1]
            self._last_closed_time_value = (
                value.item() if hasattr(value, "item") else value
            )
        else:
            self._last_closed_time_value = None
        self._time_values = []
        self._time_bounds = []
        self._saw_time_bounds = False
        self._time_offset = 0
        self._write_count = 0
        self._first_write_after_preserve = True
        self._staging.reset()

    def _time_chunk(
        self,
        data_array: np.ndarray,
        *,
        time_values: Any | None,
        time_bounds: Any | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if data_array.ndim <= self._time_dim_index:
            raise ValueError(
                f"Data for {self._ctx.variable.names()[0]!r} does not include "
                f"the expected time dimension {self._time_dim!r}."
            )
        chunk_len = int(data_array.shape[self._time_dim_index])
        if time_values is None:
            if self._initial_time_values.size == 0:
                raise ValueError(
                    "time_values must be provided when the initial time axis "
                    "does not contain complete values."
                )
            end = self._time_offset + chunk_len
            if end > self._initial_time_values.shape[0]:
                raise ValueError(
                    "The number of written time records exceeds the initial "
                    "time axis length."
                )
            values = np.asarray(self._initial_time_values[self._time_offset : end])
            if time_bounds is None and self._initial_time_bounds is not None:
                bounds = np.asarray(self._initial_time_bounds[self._time_offset : end])
            elif time_bounds is None:
                bounds = None
            else:
                bounds = bounds_as_pairs(np.asarray(time_bounds), chunk_len)
        else:
            values = np.asarray(time_values)
            if values.ndim == 0:
                values = values.reshape(1)
            elif values.ndim != 1:
                raise ValueError("time_values must be a one-dimensional array.")
            if values.shape[0] != chunk_len:
                raise ValueError(
                    f"time_values length {values.shape[0]} does not match "
                    f"the data chunk time length {chunk_len}."
                )
            bounds = (
                None
                if time_bounds is None
                else bounds_as_pairs(np.asarray(time_bounds), chunk_len)
            )
        return values, bounds

    def _axes_with_time(
        self,
        time_values: np.ndarray,
        time_bounds: np.ndarray | None,
    ) -> tuple[Axis, ...]:
        axes = list(self._ctx.axes)
        updates: dict[str, Any] = {
            "values": time_values.tolist(),
        }
        if time_bounds is not None:
            updates["bounds"] = time_bounds.tolist()
        elif self._time_axis.bounds is not None:
            updates["bounds"] = None
        axes[self._time_axis_index] = self._time_axis.updated(**updates)
        return tuple(axes)

    def _validate_chunk_shape(
        self,
        data: np.ndarray,
        chunk_time_len: int,
    ) -> None:
        expected_shape = []
        for index, dim in enumerate(self._ctx.dims):
            if index == self._time_dim_index:
                expected_shape.append(chunk_time_len)
                continue
            expected_shape.append(self._dimension_size(dim))
        actual = tuple(int(size) for size in data.shape)
        expected = tuple(expected_shape)
        if actual != expected:
            raise ValueError(
                f"Data chunk shape {actual!r} does not match expected shape "
                f"{expected!r} for dimensions {self._ctx.dims!r}."
            )

    def _dimension_size(self, dim: str) -> int:
        for axis in self._ctx.axes:
            dims = self._ctx.axis_dims.get(axis.name, ())
            if dim not in dims:
                continue
            values = axis.values_array()
            if values.ndim == 0:
                return 1
            if len(dims) == 1:
                return int(values.shape[0])
            dim_index = dims.index(dim)
            return int(values.shape[dim_index])
        raise ValueError(f"Cannot determine expected size for dimension {dim!r}.")

    def _validate_time_order(
        self,
        time_values: np.ndarray,
        time_bounds: np.ndarray | None,
    ) -> None:
        if not self._time_values:
            if (
                self._first_write_after_preserve
                and self._last_closed_time_value is not None
                and time_values.size
                and time_values[0] <= self._last_closed_time_value
            ):
                raise ValueError("Time values must be strictly monotonic across files.")
            return
        previous = self._time_values[-1]
        if previous.size and time_values.size and time_values[0] <= previous[-1]:
            raise ValueError("Time values must be strictly monotonic across writes.")
        if time_bounds is None or not self._time_bounds:
            return
        previous_bounds = self._time_bounds[-1]
        if previous_bounds.size == 0:
            return
        previous_end = previous_bounds[-1, 1]
        next_start = time_bounds[0, 0]
        if np.issubdtype(np.asarray(previous_end).dtype, np.number) and not np.isclose(
            previous_end,
            next_start,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Time bounds must be contiguous across writes within a single "
                "output file."
            )

    def _validate_zfactor_chunks(
        self,
        zfactors: Mapping[str, Any] | None,
        chunk_ctx: Any,
        chunk_time_len: int,
    ) -> dict[str, np.ndarray]:
        zfactor_values = {str(name): value for name, value in (zfactors or {}).items()}
        known = {
            str(alias)
            for zfactor in self._ctx.zfactors
            for alias in (zfactor.name, zfactor.out_name or zfactor.name)
        }
        unknown = sorted(str(name) for name in zfactor_values if str(name) not in known)
        if unknown:
            raise ValueError(f"Unknown zfactor chunk(s): {unknown!r}.")

        chunks: dict[str, np.ndarray] = {}
        for zfactor in self._ctx.zfactors:
            out_name = str(zfactor.out_name or zfactor.name)
            dims = self._zfactor_dims(zfactor)
            if self._time_dim not in dims:
                if zfactor.name in zfactor_values or out_name in zfactor_values:
                    raise ValueError(
                        f"Zfactor {out_name!r} does not include time and "
                        "cannot be supplied per write."
                    )
                continue

            name = str(zfactor.name)
            if name in zfactor_values:
                value = zfactor_values.pop(name)
            elif out_name in zfactor_values:
                value = zfactor_values.pop(out_name)
            else:
                value = None
            must_supply = (
                self._staging.has_array(out_name)
                or zfactor.values is None
                or np.asarray(zfactor.values).size == 0
            )
            if value is None:
                if must_supply:
                    raise ValueError(
                        f"Time-varying zfactor {out_name!r} must be supplied "
                        "for every write."
                    )
                continue
            if self._write_count > 0 and not self._staging.has_array(out_name):
                raise ValueError(
                    f"Time-varying zfactor {out_name!r} cannot start being "
                    "supplied after earlier writes omitted it."
                )

            data = np.asarray(value)
            self._validate_zfactor_chunk_shape(
                out_name,
                data,
                dims,
                chunk_time_len,
            )
            validate_variable_values(
                zfactor,
                chunk_ctx.axes,
                data,
                dims,
                chunk_ctx.axis_dims,
                name=out_name,
                table_id=str(zfactor.table_entry or "formula_terms"),
            )
            chunks[out_name] = data

        if zfactor_values:
            unknown = sorted(str(name) for name in zfactor_values)
            raise ValueError(f"Unknown zfactor chunk(s): {unknown!r}.")
        return chunks

    def _validate_zfactor_chunk_shape(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...],
        chunk_time_len: int,
    ) -> None:
        expected_shape = []
        for dim in dims:
            if dim == self._time_dim:
                expected_shape.append(chunk_time_len)
            else:
                expected_shape.append(self._dimension_size(dim))
        expected = tuple(expected_shape)
        actual = tuple(int(size) for size in data.shape)
        if actual != expected:
            raise ValueError(
                f"Zfactor {name!r} chunk shape {actual!r} does not match "
                f"expected shape {expected!r} for dimensions {dims!r}."
            )

    def _append_zfactors_to_zarr(self, zfactors: Mapping[str, np.ndarray]) -> None:
        for name, data in zfactors.items():
            zfactor = next(
                zfactor
                for zfactor in self._ctx.zfactors
                if str(zfactor.out_name or zfactor.name) == name
            )
            dims = self._zfactor_dims(zfactor)
            time_dim_index = dims.index(self._time_dim)
            self._staging.append(
                name,
                data,
                time_dim_index,
            )

    def _final_zfactors(self, ctx: Any) -> tuple[ZFactor, ...]:
        if not any(
            self._staging.has_array(str(zfactor.out_name or zfactor.name))
            for zfactor in ctx.zfactors
        ):
            return ctx.zfactors

        final_zfactors: list[ZFactor] = []
        final_time_len = sum(int(values.shape[0]) for values in self._time_values)
        for zfactor in ctx.zfactors:
            out_name = str(zfactor.out_name or zfactor.name)
            array = self._staging.array(out_name)
            if array is None:
                final_zfactors.append(zfactor)
                continue
            dims = self._zfactor_dims(zfactor)
            if int(array.shape[dims.index(self._time_dim)]) != final_time_len:
                raise ValueError(
                    f"Time-varying zfactor {out_name!r} was not supplied "
                    "for every write."
                )
            final_zfactors.append(
                zfactor.updated(values=self._staging.lazy_array(out_name))
            )
        return tuple(final_zfactors)

    def _zfactor_dims(self, zfactor: ZFactor) -> tuple[str, ...]:
        return _assembly.named_dimensions(zfactor.dimensions or (), self._ctx.axis_dims)

    def _final_axes(self) -> tuple[Axis, ...]:
        values = np.concatenate(self._time_values, axis=0)
        bounds = None
        if self._saw_time_bounds:
            if len(self._time_bounds) != len(self._time_values):
                raise ValueError(
                    "time_bounds must be supplied for every write or omitted "
                    "for every write."
                )
            bounds = np.concatenate(self._time_bounds, axis=0)
        return self._axes_with_time(values, bounds)


@dataclass
class _ZarrStagingStore:
    """Manage temporary Zarr arrays used by DatasetWriter."""

    root: Path
    path: Path
    _group: Any
    _arrays: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, staging_dir: str | Path | None = None) -> _ZarrStagingStore:
        root = Path(
            tempfile.mkdtemp(
                prefix="cmor-datasetwriter-",
                dir=str(staging_dir) if staging_dir is not None else None,
            )
        )
        path = root / "staging.zarr"
        return cls(root=root, path=path, _group=zarr.open_group(str(path), mode="w"))

    def append(self, name: str, data: np.ndarray, time_dim_index: int) -> Any:
        array = self._arrays.get(name)
        if array is None:
            shape = list(data.shape)
            shape[time_dim_index] = 0
            chunks = tuple(max(1, int(size)) for size in data.shape)
            create = (
                self._group.create_array
                if hasattr(self._group, "create_array")
                else self._group.create_dataset
            )
            array = create(name, shape=tuple(shape), chunks=chunks, dtype=data.dtype)

        new_shape = list(array.shape)
        start = int(new_shape[time_dim_index])
        stop = start + int(data.shape[time_dim_index])
        new_shape[time_dim_index] = stop
        array.resize(tuple(new_shape))
        index = [slice(None)] * data.ndim
        index[time_dim_index] = slice(start, stop)
        array[tuple(index)] = data
        self._arrays[name] = array
        return array

    def array(self, name: str) -> Any | None:
        return self._arrays.get(name)

    def has_array(self, name: str) -> bool:
        return name in self._arrays

    def lazy_array(self, name: str) -> Any:
        array = self.array(name)
        if array is None:
            raise ValueError(f"No staged Zarr array exists for {name!r}.")
        data = da.from_zarr(array)
        byteorder = getattr(data.dtype, "byteorder", None)
        if byteorder not in (None, "=", "|"):
            dtype = data.dtype.newbyteorder("=")
            data = data.map_blocks(
                lambda block: block.astype(dtype, copy=False),
                meta=np.array([], dtype=dtype),
            )
        return data

    def reset(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        self._group = zarr.open_group(str(self.path), mode="w")
        self._arrays.clear()

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._arrays.clear()
