"""Registry for DP synthesizers."""

from typing import Dict, Type

from fairness_auditbench.synthesizers.base import BaseSynthesizer

_SYNTHESIZER_REGISTRY: Dict[str, Type[BaseSynthesizer]] = {}


def register_synthesizer(name: str):
    """Decorator to register a synthesizer class by name."""

    def wrapper(cls: Type[BaseSynthesizer]):
        _SYNTHESIZER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_synthesizer(name: str, **kwargs) -> BaseSynthesizer:
    """Instantiate a synthesizer by name.

    Args:
        name: Name of the registered synthesizer.
        **kwargs: Extra arguments to pass to the synthesizer constructor.

    Returns:
        BaseSynthesizer instance.
    """
    if name not in _SYNTHESIZER_REGISTRY:
        available = list(_SYNTHESIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown synthesizer '{name}'. Available: {available}"
        )
    return _SYNTHESIZER_REGISTRY[name](**kwargs)
