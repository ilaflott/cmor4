"""CMOR determines tracking_id; any user-supplied value is replaced."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import cmor
from table_helpers import cmip7_project

_IGNORED = "tracking_id supplied in .* is ignored"
_USER_ID = "user-supplied-id"


def _base(outpath: Path) -> dict[str, Any]:
    return {
        "activity_id": "CMIP",
        "experiment_id": "amip",
        "forcing_index": "f1",
        "grid_label": "g101",
        "initialization_index": "i1",
        "institution_id": "CNRM-CERFACS",
        "license_id": "CC-BY-4.0",
        "nominal_resolution": "100 km",
        "physics_index": "p1",
        "realization_index": "r1",
        "region": "glb",
        "source_id": "CNRM-ESM2-1e",
        "outpath": str(outpath),
    }


@pytest.fixture(scope="module")
def project():
    return cmip7_project("tables/CMIP7_ocean.json")


def _axes(project, *, time_values: bool = True):
    time_kwargs: dict[str, Any] = {}
    if time_values:
        time_kwargs = {"values": [15.0, 45.0], "bounds": [[0.0, 30.0], [30.0, 60.0]]}
    return [
        project.axis(
            "time",
            calendar="360_day",
            frequency="mon",
            units="days since 1979-01-01",
            **time_kwargs,
        ),
        project.axis(
            "latitude",
            values=[10.0, 20.0, 30.0],
            bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
        ),
        project.axis(
            "longitude",
            values=[0.0, 90.0, 180.0, 270.0],
            bounds=[[-45.0, 45.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]],
        ),
    ]


def _cmorize(project, dataset, path: Path, **kwargs: Any) -> str:
    out = cmor.cmorize(
        dataset,
        project.variable("tos_tavg-u-hxy-sea", table_id="ocean"),
        _axes(project),
        np.full((2, 3, 4), 20.0, dtype="f4"),
        path=path,
        **kwargs,
    )
    with xr.open_dataset(out) as ds:
        return ds.attrs["tracking_id"]


def _assert_cmip7_id(tracking_id: str) -> None:
    prefix, _, identifier = tracking_id.partition("/")
    assert prefix == "hdl:21.14107"
    assert str(uuid.UUID(identifier)) == identifier


def test_generated_id_matches_cv(project, tmp_path):
    tracking_id = _cmorize(
        project, project.dataset_info(_base(tmp_path)), tmp_path / "a.nc"
    )
    _assert_cmip7_id(tracking_id)


def test_each_file_gets_a_new_id(project, tmp_path):
    dataset = project.dataset_info(_base(tmp_path))
    first = _cmorize(project, dataset, tmp_path / "a.nc")
    second = _cmorize(project, dataset, tmp_path / "b.nc")
    assert first != second


def test_dataset_info_tracking_id_overwritten(project, tmp_path):
    with pytest.warns(UserWarning, match=_IGNORED):
        dataset = project.dataset_info({**_base(tmp_path), "tracking_id": _USER_ID})
    tracking_id = _cmorize(project, dataset, tmp_path / "a.nc")
    assert tracking_id != _USER_ID
    _assert_cmip7_id(tracking_id)


def test_wrapped_dataset_info_tracking_id_overwritten(project, tmp_path):
    with pytest.warns(UserWarning, match=_IGNORED):
        dataset = project.dataset_info(
            cmor.DatasetInfo(**_base(tmp_path), tracking_id=_USER_ID)
        )
    assert _cmorize(project, dataset, tmp_path / "a.nc") != _USER_ID


def test_cmorize_attrs_tracking_id_overwritten(project, tmp_path):
    dataset = project.dataset_info(_base(tmp_path))
    with pytest.warns(UserWarning, match=_IGNORED):
        tracking_id = _cmorize(
            project, dataset, tmp_path / "a.nc", attrs={"tracking_id": _USER_ID}
        )
    assert tracking_id != _USER_ID
    _assert_cmip7_id(tracking_id)


def test_writer_attrs_tracking_id_overwritten(project, tmp_path):
    with pytest.warns(UserWarning, match=_IGNORED):
        with cmor.DatasetWriter(
            project.dataset_info(_base(tmp_path)),
            project.variable("tos_tavg-u-hxy-sea", table_id="ocean"),
            _axes(project, time_values=False),
            attrs={"tracking_id": _USER_ID},
            path=tmp_path / "writer.nc",
        ) as writer:
            writer.write(
                np.full((1, 3, 4), 20.0, dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )
    with xr.open_dataset(writer.path) as ds:
        tracking_id = ds.attrs["tracking_id"]
    assert tracking_id != _USER_ID
    _assert_cmip7_id(tracking_id)


def test_no_project_tracking_id_dropped(tmp_path):
    time = cmor.Axis(
        name="time",
        standard_name="time",
        axis="T",
        values=[0.0, 6.0],
        units="hours since 2020-01-01",
        calendar="standard",
        frequency="6hr",
    )
    variable = cmor.Variable(name="tas", dimensions=("time",), units="K")
    with pytest.warns(UserWarning, match=_IGNORED):
        out = cmor.cmorize(
            cmor.DatasetInfo(tracking_id=_USER_ID),
            variable,
            [time],
            np.full(2, 280.0),
            path=tmp_path / "np.nc",
        )
    with xr.open_dataset(out) as ds:
        assert "tracking_id" not in ds.attrs
