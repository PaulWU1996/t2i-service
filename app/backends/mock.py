"""CPU-only fake backend for development and tests (no torch needed)."""
import time

from PIL import Image, ImageDraw

from ..config import Settings
from ..models import ModelSpec
from .base import GenParams


class MockBackend:
    def __init__(self, spec: ModelSpec, settings: Settings):
        self._delay = settings.mock_delay_seconds

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, p: GenParams) -> Image.Image:
        if self._delay:
            time.sleep(self._delay)
        if "FAIL" in p.prompt:  # test hook for the failure path
            raise RuntimeError("mock backend forced failure")
        colour = (p.seed % 200 + 30, (p.seed // 7) % 200 + 30, (p.seed // 49) % 200 + 30)
        img = Image.new("RGB", (p.width, p.height), colour)
        ImageDraw.Draw(img).text((10, 10), p.prompt[:80], fill=(255, 255, 255))
        return img
