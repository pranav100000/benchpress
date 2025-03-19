"""Exception classes for the benchpress package.

This module defines a hierarchy of exceptions used throughout the benchpress package
to provide more specific error information and enable better error handling.
"""

class BenchpressError(Exception):
    """Base exception for all Benchpress errors.

    All custom exceptions in the benchpress package should inherit from this class.
    This allows for easy catching of all Benchpress-specific errors.
    """
    pass


class DatasetError(BenchpressError):
    """Error related to dataset loading or processing.

    Raised when there's an issue with dataset operations such as:
    - Dataset file not found
    - Invalid dataset format
    - Problems accessing dataset content
    - Authentication issues with dataset providers
    """
    pass


class ModelError(BenchpressError):
    """Error related to model invocation or processing.

    Raised when there's an issue with model operations such as:
    - API authentication failures
    - Rate limiting or quota exhaustion
    - Timeout during model inference
    - Invalid responses from model APIs
    """
    pass


class TaskError(BenchpressError):
    """Error related to task definition or execution.

    Raised when there's an issue with task operations such as:
    - Invalid task configuration
    - Problems during task execution
    - Task registration issues
    """
    pass


class ExtractionError(BenchpressError):
    """Error related to answer extraction.

    Raised when there's an issue with extracting answers from model responses such as:
    - No answer found in response
    - Multiple conflicting answers found
    - Invalid format of extracted answer
    """
    pass
