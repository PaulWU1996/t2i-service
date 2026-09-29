"""Generic diffusers text-to-image backend.

Which pipeline runs is decided by ModelSpec.pipeline_class (e.g. Flux2KleinPipeline,
Krea2Pipeline), so a new model is a registry entry, not new code. torch and
diffusers are imported lazily so the API layer can run without them (tests, mock).
"""
import gc
import logging
from pathlib import Path

from PIL import Image

from ..config import Settings
from ..models import ModelSpec
from .base import GenParams

log = logging.getLogger("t2i.backend")


class DiffusersBackend:
    def __init__(self, spec: ModelSpec, settings: Settings):
        self.spec = spec
        self.settings = settings
        self._pipe = None

    def _source(self) -> str:
        local = Path(self.settings.models_dir) / self.spec.name
        if local.is_dir():
            log.info("loading %s from local path %s", self.spec.name, local)
            return str(local)
        log.info("loading %s from Hugging Face repo %s", self.spec.name, self.spec.repo_id)
        return self.spec.repo_id

    def load(self) -> None:
        import diffusers
        import torch

        pipeline_cls = getattr(diffusers, self.spec.pipeline_class)
        pipe = pipeline_cls.from_pretrained(self._source(), torch_dtype=torch.bfloat16)
        if self.settings.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe

    def unload(self) -> None:
        import torch

        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, p: GenParams) -> Image.Image:
        import torch

        generator = torch.Generator(device="cuda").manual_seed(p.seed)
        try:
            with torch.inference_mode():
                out = self._pipe(
                    prompt=p.prompt,
                    height=p.height,
                    width=p.width,
                    num_inference_steps=p.steps,
                    guidance_scale=p.guidance,
                    generator=generator,
                )
            return out.images[0]
        finally:
            torch.cuda.empty_cache()
