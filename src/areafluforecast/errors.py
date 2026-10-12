"""The exception and warning types the package raises."""


class DataError(ValueError):
    """The input data cannot be used as given. The message says what to fix."""


class DataWarning(UserWarning):
    """The input data can be used, but something in it deserves a look."""


class TrustWarning(UserWarning):
    """A forecast comes from a model that has not shown it beats persistence here."""
