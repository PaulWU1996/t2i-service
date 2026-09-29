from ..config import Settings
from ..models import ModelSpec
from .base import Backend, GenParams

__all__ = ["Backend", "GenParams", "build_backend"]


def build_backend(spec: ModelSpec, settings: Settings) -> Backend:
    if spec.backend == "mock":
        from .mock import MockBackend

        return MockBackend(spec, settings)
    if spec.backend == "diffusers":
        from .diffusers_backend import DiffusersBackend

        return DiffusersBackend(spec, settings)
    raise ValueError(f"unknown backend '{spec.backend}' for model '{spec.name}'")
