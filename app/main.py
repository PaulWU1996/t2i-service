"""HTTP layer. Run with:  uvicorn app.main:create_app --factory --port 8000"""
import asyncio
import logging
import queue
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse

from . import __version__
from .config import Settings
from .jobs import Job, JobStore
from .manager import ModelManager
from .models import build_registry
from .schemas import GenerationRequest, Health, JobView, ModelInfo
from .storage import ID_RE, Storage
from .worker import Worker

log = logging.getLogger("t2i")

TERMINAL = ("succeeded", "failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = build_registry(settings)
        if settings.default_model not in registry:
            raise RuntimeError(
                f"T2I_DEFAULT_MODEL '{settings.default_model}' is not one of {sorted(registry)}"
            )
        storage = Storage(settings.output_dir)
        storage.sweep(settings.retention_days)
        store = JobStore()
        manager = ModelManager(settings, registry)
        q: "queue.Queue[str | None]" = queue.Queue(maxsize=settings.max_queue_size)
        worker = Worker(q, store, manager, storage, settings)
        worker.start()

        app.state.settings, app.state.registry = settings, registry
        app.state.storage, app.state.store = storage, store
        app.state.manager, app.state.queue, app.state.worker = manager, q, worker

        if settings.preload_model:
            def _preload() -> None:
                try:
                    manager.get(settings.default_model)
                except Exception:
                    log.exception("preload of %s failed (will retry on first job)", settings.default_model)

            threading.Thread(target=_preload, name="t2i-preload", daemon=True).start()

        log.info("t2i service %s ready; default model %s; models %s",
                 __version__, settings.default_model, sorted(registry))
        yield
        worker.stop()
        worker.join(timeout=5)
        manager.shutdown()

    app = FastAPI(title="T2I Service", version=__version__, lifespan=lifespan)

    def _view(meta: dict) -> JobView:
        view = JobView.model_validate(meta)
        if view.status == "succeeded":
            view.image_url = f"/v1/generations/{view.id}/image"
        return view

    def _find_meta(job_id: str) -> dict | None:
        if not ID_RE.match(job_id):
            return None
        job = app.state.store.get(job_id)
        if job:
            return job.to_meta()
        return app.state.storage.load_meta(job_id)  # survives restarts

    # ------------------------------------------------------------------ routes

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        return Health(
            status="ok",
            loaded_model=app.state.manager.loaded_model,
            busy=app.state.worker.current_job is not None,
            queue_depth=app.state.queue.qsize(),
        )

    @app.get("/v1/models", response_model=list[ModelInfo])
    async def list_models() -> list[ModelInfo]:
        return [
            ModelInfo(
                name=s.name, default=s.name == settings.default_model, license=s.license,
                default_width=s.default_width, default_height=s.default_height,
                default_steps=s.default_steps, default_guidance=s.default_guidance,
            )
            for s in app.state.registry.values()
        ]

    @app.post("/v1/generations", response_model=JobView, status_code=202)
    async def create_generation(
        req: GenerationRequest,
        response: Response,
        wait: bool = Query(False, description="Hold the connection until the job finishes (or timeout)."),
        timeout: float | None = Query(None, ge=0, description="Max seconds to wait when wait=true."),
    ) -> JobView:
        name = req.model or settings.default_model
        spec = app.state.registry.get(name)
        if spec is None:
            raise HTTPException(400, f"unknown model '{name}'; available: {sorted(app.state.registry)}")
        if len(req.prompt) > settings.max_prompt_chars:
            raise HTTPException(422, f"prompt exceeds {settings.max_prompt_chars} characters")

        width = req.width if req.width is not None else spec.default_width
        height = req.height if req.height is not None else spec.default_height
        if max(width, height) > settings.max_side or width * height > settings.max_pixels:
            raise HTTPException(
                422, f"image too large: max side {settings.max_side}px, max {settings.max_pixels} pixels"
            )

        job = Job(
            id=uuid.uuid4().hex,
            model=name,
            prompt=req.prompt,
            width=width,
            height=height,
            steps=req.steps if req.steps is not None else spec.default_steps,
            guidance=req.guidance if req.guidance is not None else spec.default_guidance,
            seed=req.seed if req.seed is not None else secrets.randbelow(2**32),
        )
        app.state.store.add(job)
        try:
            app.state.queue.put_nowait(job.id)
        except queue.Full:
            app.state.store.remove(job.id)
            raise HTTPException(
                429, "queue is full; retry shortly", headers={"Retry-After": "10"}
            ) from None
        log.info("job %s queued model=%s %dx%d steps=%d prompt_chars=%d",
                 job.id, name, width, height, job.steps, len(job.prompt))

        if wait:
            limit = min(timeout if timeout is not None else settings.default_wait_seconds,
                        settings.max_wait_seconds)
            deadline = time.monotonic() + limit
            while time.monotonic() < deadline:
                current = app.state.store.get(job.id)
                if current and current.status in TERMINAL:
                    response.status_code = 200
                    return _view(current.to_meta())
                await asyncio.sleep(0.2)
        current = app.state.store.get(job.id) or job
        if current.status in TERMINAL:
            response.status_code = 200
        return _view(current.to_meta())

    @app.get("/v1/generations/{job_id}", response_model=JobView)
    async def get_generation(job_id: str) -> JobView:
        meta = _find_meta(job_id)
        if meta is None:
            raise HTTPException(404, "generation not found")
        return _view(meta)

    @app.get("/v1/generations/{job_id}/image",
             responses={200: {"content": {"image/png": {}}}})
    async def get_image(job_id: str) -> FileResponse:
        meta = _find_meta(job_id)
        if meta is None:
            raise HTTPException(404, "generation not found")
        if meta["status"] in ("queued", "running"):
            raise HTTPException(409, f"generation is {meta['status']}; image not ready")
        if meta["status"] == "failed":
            raise HTTPException(404, "generation failed; no image")
        path = app.state.storage.image_path(job_id)
        if not path.exists():
            raise HTTPException(404, "image no longer available (retention)")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "private, max-age=3600"})

    return app
