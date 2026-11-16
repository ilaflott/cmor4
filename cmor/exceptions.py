class AxisValidationError(ValueError):
    """Coordinate values or bounds are invalid.

    For example: values that are not monotonic or are outside the table's
    valid range, bounds that overlap or have the wrong shape, missing bounds
    the table requires, an unknown calendar, or time steps that do not
    match the frequency.
    """


class ControlledVocabularyError(ValueError):
    """A global attribute is missing or not allowed by the project CV.

    For example: a required attribute is missing, a value such as
    ``experiment_id`` is not in the CV, or two attributes disagree (such as
    ``source_type`` and the experiment's required source types).

    See :doc:`/usage/controlled_vocabularies` for the checks CMOR runs.
    """


class TableValidationError(ValueError):
    """Metadata does not match the project tables.

    For example: a variable or coordinate name is not in the tables, a name
    is in more than one table and no ``table_id`` was given, or a value you
    supplied (such as ``units`` or ``dimensions``) differs from the table.
    """


class VariableValidationError(ValueError):
    """Data values fail the variable's checks.

    For example: the data contains NaN values that are not marked as
    missing, or the mean absolute value is more than 10 times outside the
    table's ``ok_min_mean_abs``/``ok_max_mean_abs`` range. Values outside
    ``valid_min``/``valid_max`` produce a warning, not this error.
    """
