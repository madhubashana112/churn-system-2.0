"""Safe provider failures shared across application and presentation layers."""


class AIServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503, code: str = "AI_UNAVAILABLE"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
