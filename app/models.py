"""Model registry. Adding a model that diffusers supports is one more ModelSpec."""
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class ModelSpec:
    name: str  # public name used in the API and as the local-weights folder name
    backend: str  # "diffusers" | "mock"
    repo_id: str  # Hugging Face repo (ignored by mock)
    pipeline_class: str  # attribute name on the diffusers module
    default_steps: int
    default_guidance: float
    default_width: int = 1024
    default_height: int = 1024
    license: str = ""
    # Whether the underlying model can also do image editing. Not exposed by
    # the API yet; recorded so a future /v1/edits endpoint can check it.
    supports_edit: bool = False


def build_registry(settings: Settings) -> dict[str, ModelSpec]:
    specs = [
        ModelSpec(
            name="flux2-klein-4b",
            backend="diffusers",
            repo_id="black-forest-labs/FLUX.2-klein-4B",
            pipeline_class="Flux2KleinPipeline",
            default_steps=4,  # step-distilled
            default_guidance=1.0,  # guidance-distilled
            license="Apache-2.0",
            supports_edit=True,
        ),
    ]
    if settings.enable_mock_model:
        specs.append(
            ModelSpec(
                name="mock",
                backend="mock",
                repo_id="",
                pipeline_class="",
                default_steps=1,
                default_guidance=0.0,
                default_width=512,
                default_height=512,
                license="n/a",
            )
        )
    return {s.name: s for s in specs}
