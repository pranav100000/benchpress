"""Registry for answer extractors."""

from typing import Callable, Dict, Optional, Type

from .base import BaseExtractor


# Registry of extractors
extractor_registry: Dict[str, Type[BaseExtractor]] = {}


def register_extractor(name: Optional[str] = None) -> Callable:
    """Register an extractor class.

    Args:
        name: Optional name for the extractor. If not provided, the class name will be used.

    Returns:
        A decorator that registers the class
    """
    def decorator(cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
        extractor_name = name or cls.__name__
        extractor_registry[extractor_name] = cls
        return cls
    
    return decorator


def get_extractor(name: str) -> Optional[Type[BaseExtractor]]:
    """Get an extractor by name.

    Args:
        name: The name of the extractor

    Returns:
        The extractor class, or None if not found
    """
    return extractor_registry.get(name)