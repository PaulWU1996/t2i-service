"""Keeps at most one model in GPU memory; switches on demand."""
import logging
import threading

from .backends import Backend, build_backend
from .config import Settings
from .models import ModelSpec

log = logging.getLogger("t2i.manager")


class ModelManager:
    def __init__(self, settings: Settings, registry: dict[str, ModelSpec]):
        self._settings = settings
        self._registry = registry
        self._lock = threading.Lock()
        self._backend: Backend | None = None
        self.loaded_model: str | None = None

    def get(self, name: str) -> Backend:
        with self._lock:
            if self.loaded_model != name:
                self._unload_locked()
                log.info("loading model %s", name)
                backend = build_backend(self._registry[name], self._settings)
                backend.load()
                self._backend, self.loaded_model = backend, name
                log.info("model %s ready", name)
            return self._backend  # type: ignore[return-value]

    def _unload_locked(self) -> None:
        if self._backend is not None:
            log.info("unloading model %s", self.loaded_model)
            self._backend.unload()
        self._backend, self.loaded_model = None, None

    def shutdown(self) -> None:
        with self._lock:
            self._unload_locked()
