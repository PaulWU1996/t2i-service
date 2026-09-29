from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class GenParams:
    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int


class Backend(Protocol):
    """Everything the service needs from a model. Keep diffusers types out of here."""

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(self, params: GenParams) -> Image.Image: ...
