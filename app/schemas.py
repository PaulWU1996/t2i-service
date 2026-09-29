from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# FLUX.2 (and most DiT models) need dimensions divisible by 16.
SIZE_MULTIPLE = 16


class GenerationRequest(BaseModel):
    # Unknown fields are rejected (422) rather than silently ignored, so a
    # client sending e.g. "image" before image-to-image exists gets a clear error.
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="Natural-language description of the image.")
    model: str | None = Field(default=None, description="Model name; see GET /v1/models.")
    width: int | None = Field(default=None, ge=256)
    height: int | None = Field(default=None, ge=256)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be blank")
        return v

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, v: int | None) -> int | None:
        if v is not None and v % SIZE_MULTIPLE:
            raise ValueError(f"must be a multiple of {SIZE_MULTIPLE}")
        return v


class JobParams(BaseModel):
    width: int
    height: int
    steps: int
    guidance: float
    seed: int


class Timings(BaseModel):
    queue_seconds: float | None = None
    load_seconds: float | None = None
    generate_seconds: float | None = None


class JobView(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    model: str
    prompt: str
    params: JobParams
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    timings: Timings | None = None
    error: str | None = None
    image_url: str | None = None


class ModelInfo(BaseModel):
    name: str
    default: bool
    license: str
    default_width: int
    default_height: int
    default_steps: int
    default_guidance: float


class Health(BaseModel):
    status: str
    loaded_model: str | None
    busy: bool
    queue_depth: int
