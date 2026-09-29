"""Service configuration. Every setting is an environment variable prefixed T2I_."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="T2I_", extra="ignore")

    # --- models -----------------------------------------------------------
    default_model: str = "flux2-klein-4b"
    # If <models_dir>/<model-name>/ exists, weights load from there instead of
    # the Hugging Face Hub (for pre-provisioned weights on a persistent disk).
    models_dir: Path = Path("/data/models")
    preload_model: bool = False  # load the default model at startup
    cpu_offload: bool = False  # enable_model_cpu_offload() instead of .to("cuda")
    enable_mock_model: bool = False  # CPU-only fake model for dev / CI

    # --- storage ----------------------------------------------------------
    output_dir: Path = Path("/data/outputs")
    retention_days: int = 30  # 0 disables cleanup

    # --- limits -----------------------------------------------------------
    max_queue_size: int = 32  # jobs waiting (not counting the one running)
    max_side: int = 2048
    max_pixels: int = 2048 * 1152  # ~2.36 MP
    max_prompt_chars: int = 4000
    default_wait_seconds: float = 60.0  # ?wait=true without ?timeout=
    max_wait_seconds: float = 90.0  # keep below proxy limits (Cloudflare: 100 s)

    # --- misc -------------------------------------------------------------
    log_level: str = "INFO"
    mock_delay_seconds: float = 0.0  # mock backend only (tests)
