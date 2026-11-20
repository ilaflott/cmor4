"""Bare-number RIPF indices get their letter only when the CV requires it."""

from __future__ import annotations

from typing import Any

import pytest

from cmor import ControlledVocabulary, DatasetInfo
from cmor.exceptions import ControlledVocabularyError
from table_helpers import cmip7_project

from test_cmip6_datasets import (
    CMIP6_CV_FILE,
    CMIP6_TABLE_ROOT,
    _amip_attrs,
    _cmip6_project,
)

_CMIP7_BASE: dict[str, Any] = {
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
}


@pytest.fixture(scope="module")
def cmip7():
    project = cmip7_project("tables/CMIP7_atmos.json")
    return project, project.variable("tas_tavg-h2m-hxy-u")


def _cmip7_attrs(cmip7, **overrides: Any) -> dict[str, Any]:
    project, variable = cmip7
    info = project.dataset_info({**_CMIP7_BASE, **overrides})
    return info.global_attributes(variable)


@pytest.mark.parametrize("value", [1, "1"])
def test_cmip7_bare_initialization_index_gets_prefix(cmip7, value):
    attrs = _cmip7_attrs(cmip7, initialization_index=value)
    assert attrs["initialization_index"] == "i1"
    assert attrs["variant_label"] == "r1i1p1f1"


def test_cmip7_bare_rpf_indices_get_prefix(cmip7):
    attrs = _cmip7_attrs(cmip7, realization_index=3, physics_index=2, forcing_index="4")
    assert attrs["realization_index"] == "r3"
    assert attrs["physics_index"] == "p2"
    assert attrs["forcing_index"] == "f4"
    assert attrs["variant_label"] == "r3i1p2f4"


def test_cmip7_prefixed_index_unchanged(cmip7):
    attrs = _cmip7_attrs(cmip7, initialization_index="i2")
    assert attrs["initialization_index"] == "i2"
    assert attrs["variant_label"] == "r1i2p1f1"


def test_cmip7_zero_index_still_rejected(cmip7):
    with pytest.raises(ValueError, match="out of the valid range"):
        _cmip7_attrs(cmip7, initialization_index=0)


def test_cmip6_bare_indices_unchanged(tmp_path):
    if not (CMIP6_TABLE_ROOT / CMIP6_CV_FILE).exists():
        pytest.skip("cmip6-cmor-tables submodule not initialised")
    project = _cmip6_project()
    info = project.dataset_info(_amip_attrs(tmp_path))
    values = info.to_dict()
    assert values["realization_index"] == "3"
    assert values["initialization_index"] == "1"


# Date-style initialization indices with a letter suffix (CMIP7 pattern 2).


@pytest.mark.parametrize("value", ["i202001a", "i202001abc"])
def test_cmip7_suffixed_initialization_index_accepted(cmip7, value):
    attrs = _cmip7_attrs(cmip7, initialization_index=value)
    assert attrs["initialization_index"] == value
    assert attrs["variant_label"] == f"r1{value}p1f1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"initialization_index": "i202001f"},  # suffix letter not in a-e
        {"initialization_index": "i1a"},  # suffix needs the 6-digit date form
        {"realization_index": "r1a"},  # CMIP7 r pattern has no suffix
    ],
)
def test_cmip7_disallowed_suffix_rejected(cmip7, overrides):
    with pytest.raises(ControlledVocabularyError):
        _cmip7_attrs(cmip7, **overrides)


def test_cmip7_suffixed_parent_variant_label_accepted(cmip7):
    project, _ = cmip7
    project.cv._validate_parent_attributes({
        "experiment_id": "1pctCO2",
        "parent_experiment_id": "piControl",
        "parent_activity_id": "CMIP",
        "parent_source_id": "CNRM-ESM2-1e",
        "parent_mip_era": "CMIP7",
        "parent_time_units": "days since 1850-01-01",
        "parent_variant_label": "r1i202001ap1f1",
        "branch_time_in_child": 0.0,
        "branch_time_in_parent": 0.0,
    })


def test_suffix_rejected_without_cv_index_patterns():
    cv = ControlledVocabulary({"variant_label": ["r1i1p1f1"]})
    with pytest.raises(ControlledVocabularyError, match="letter suffix"):
        cv.validate_dataset_info(
            DatasetInfo(initialization_index="i202001a"), check_required=False
        )


def test_cmip6_suffixed_variant_label_rejected():
    if not (CMIP6_TABLE_ROOT / CMIP6_CV_FILE).exists():
        pytest.skip("cmip6-cmor-tables submodule not initialised")
    cv = _cmip6_project().cv
    with pytest.raises(ControlledVocabularyError):
        cv.validate_dataset_info(
            DatasetInfo(variant_label="r1i202001ap1f1"), check_required=False
        )
