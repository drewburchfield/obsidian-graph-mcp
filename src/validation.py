"""
Input validation for MCP tool parameters.

Provides centralized validation with:
- Type checking
- Range validation
- Default value application
- Descriptive error messages
"""

from typing import Any

from loguru import logger


class ValidationError(Exception):
    """Raised when parameter validation fails."""

    pass


def validate_required_string(
    args: dict[str, Any], param_name: str, allow_empty: bool = False, max_length: int = 10000
) -> str:
    """
    Validate required string parameter.

    Args:
        args: Argument dictionary
        param_name: Parameter name
        allow_empty: Whether to allow empty strings
        max_length: Maximum string length

    Returns:
        Validated string value

    Raises:
        ValidationError: If validation fails
    """
    if param_name not in args:
        raise ValidationError(f"Required parameter '{param_name}' is missing")

    value = args[param_name]

    if not isinstance(value, str):
        raise ValidationError(
            f"Parameter '{param_name}' must be string, got {type(value).__name__}"
        )

    if not allow_empty and len(value.strip()) == 0:
        raise ValidationError(f"Parameter '{param_name}' cannot be empty")

    if len(value) > max_length:
        raise ValidationError(
            f"Parameter '{param_name}' exceeds maximum length of {max_length} characters"
        )

    return value


def validate_int_range(
    args: dict[str, Any], param_name: str, default: int, min_val: int, max_val: int
) -> int:
    """
    Validate integer parameter with range checking.

    Args:
        args: Argument dictionary
        param_name: Parameter name
        default: Default value if not provided
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated integer (clamped to range)

    Raises:
        ValidationError: If value is out of range
    """
    if param_name not in args:
        return default

    value = args[param_name]

    # Try to convert to int
    try:
        value_int = int(value)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid type for '{param_name}': {type(value).__name__}, using default {default}"
        )
        return default

    # Range check
    if value_int < min_val or value_int > max_val:
        raise ValidationError(
            f"Parameter '{param_name}' must be in range [{min_val}, {max_val}], got {value_int}"
        )

    return value_int


def validate_float_range(
    args: dict[str, Any], param_name: str, default: float, min_val: float, max_val: float
) -> float:
    """
    Validate float parameter with range checking.

    Args:
        args: Argument dictionary
        param_name: Parameter name
        default: Default value if not provided
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated float (clamped to range)

    Raises:
        ValidationError: If value is out of range
    """
    if param_name not in args:
        return default

    value = args[param_name]

    # Try to convert to float
    try:
        value_float = float(value)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid type for '{param_name}': {type(value).__name__}, using default {default}"
        )
        return default

    # Range check
    if value_float < min_val or value_float > max_val:
        raise ValidationError(
            f"Parameter '{param_name}' must be in range [{min_val}, {max_val}], got {value_float}"
        )

    return value_float


# Tool-specific validation functions


def validate_search_notes_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate parameters for search_notes tool.

    Required:
        - query (str): Search query

    Optional:
        - limit (int): 1-50, default 10
        - threshold (float): 0.0-1.0, default 0.5

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    return {
        "query": validate_required_string(args, "query"),
        "limit": validate_int_range(args, "limit", default=10, min_val=1, max_val=50),
        "threshold": validate_float_range(args, "threshold", default=0.5, min_val=0.0, max_val=1.0),
    }


def validate_similar_notes_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate parameters for get_similar_notes tool.

    Required:
        - note_path (str): Path to source note

    Optional:
        - limit (int): 1-50, default 10
        - threshold (float): 0.0-1.0, default 0.5

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    return {
        "note_path": validate_required_string(args, "note_path"),
        "limit": validate_int_range(args, "limit", default=10, min_val=1, max_val=50),
        "threshold": validate_float_range(args, "threshold", default=0.5, min_val=0.0, max_val=1.0),
    }


def validate_connection_graph_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate parameters for get_connection_graph tool.

    Required:
        - note_path (str): Starting note path

    Optional:
        - depth (int): 1-5, default 3
        - max_per_level (int): 1-10, default 5
        - threshold (float): 0.0-1.0, default 0.5

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    return {
        "note_path": validate_required_string(args, "note_path"),
        "depth": validate_int_range(args, "depth", default=3, min_val=1, max_val=5),
        "max_per_level": validate_int_range(
            args, "max_per_level", default=5, min_val=1, max_val=10
        ),
        "threshold": validate_float_range(args, "threshold", default=0.5, min_val=0.0, max_val=1.0),
    }


def validate_hub_notes_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate parameters for get_hub_notes tool.

    All optional:
        - min_connections (int): >=1, default 10
        - threshold (float): 0.0-1.0, default 0.5
        - limit (int): 1-50, default 20

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    return {
        "min_connections": validate_int_range(
            args, "min_connections", default=10, min_val=1, max_val=1000
        ),
        "threshold": validate_float_range(args, "threshold", default=0.5, min_val=0.0, max_val=1.0),
        "limit": validate_int_range(args, "limit", default=20, min_val=1, max_val=50),
    }


def validate_orphaned_notes_args(args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate parameters for get_orphaned_notes tool.

    All optional:
        - max_connections (int): >=0, default 2
        - threshold (float): 0.0-1.0, default 0.5
        - limit (int): 1-50, default 20

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If validation fails
    """
    return {
        "max_connections": validate_int_range(
            args, "max_connections", default=2, min_val=0, max_val=100
        ),
        "threshold": validate_float_range(args, "threshold", default=0.5, min_val=0.0, max_val=1.0),
        "limit": validate_int_range(args, "limit", default=20, min_val=1, max_val=50),
    }
