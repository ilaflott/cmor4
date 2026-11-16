"""Prepared dataset metadata record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import Field

from ._metadata import DatasetMetadata


class DatasetInfo(DatasetMetadata):
    """Dataset metadata with CV defaults applied, ready for writing.

    Returned by :meth:`ProjectTables.dataset_info`; pass it to
    :func:`cmorize` or :class:`DatasetWriter`. Read fields as attributes
    (``info.source_id``). Use ``info.to_dict()`` to get all global
    attributes, including project-specific ones, as a dictionary.

    See :doc:`/usage/controlled_vocabularies` for how the defaults are
    filled.
    """

    project: Any = Field(default=None, exclude=True, repr=False)
    user_info: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @classmethod
    def from_prepared(
        cls,
        values: DatasetMetadata,
        *,
        project: Any = None,
        user_info: Mapping[str, Any] | None = None,
    ) -> DatasetInfo:
        """Create prepared dataset metadata with project context attached."""

        if not isinstance(values, DatasetMetadata):
            values = DatasetMetadata.from_mapping(cast(Mapping[str, Any], values))

        data = values.to_dict()
        data["project"] = project
        data["user_info"] = (
            dict(user_info) if user_info is not None else values.to_dict()
        )
        return cls.model_validate(data)
