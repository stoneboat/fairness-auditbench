"""Synthesizer registry and base classes."""

from .base import BaseSynthesizer
from .registry import get_synthesizer, register_synthesizer
from . import dpctgan  # ensure registration
from . import ctgan  # ensure registration

__all__ = ["BaseSynthesizer", "get_synthesizer", "register_synthesizer"]
