"""User-facing error type: the UI shows ``message`` and ``hint`` instead of a traceback."""


class AssistantError(Exception):
    """A problem the user can fix (bad input, missing artifact, wrong mode for the image)."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message
