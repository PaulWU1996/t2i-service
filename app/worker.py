"""Single worker thread: one GPU job at a time, pulled from a bounded queue."""
import logging
import queue
import threading
import time
from dataclasses import replace

from .backends import GenParams
from .config import Settings
from .jobs import Job, JobStore
from .manager import ModelManager
from .storage import Storage

log = logging.getLogger("t2i.worker")

SWEEP_INTERVAL_SECONDS = 3600


class Worker(threading.Thread):
    def __init__(self, q: "queue.Queue[str | None]", store: JobStore, manager: ModelManager,
                 storage: Storage, settings: Settings):
        super().__init__(name="t2i-worker", daemon=True)
        self.q, self.store, self.manager = q, store, manager
        self.storage, self.settings = storage, settings
        self.current_job: str | None = None
        self._last_sweep = 0.0

    def stop(self) -> None:
        self.q.put(None)  # sentinel; blocks only if the queue is full, which is fine at shutdown

    def run(self) -> None:
        while True:
            self._maybe_sweep()
            try:
                job_id = self.q.get(timeout=SWEEP_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            if job_id is None:
                return
            self._run_job(job_id)

    def _maybe_sweep(self) -> None:
        if time.monotonic() - self._last_sweep >= SWEEP_INTERVAL_SECONDS:
            self._last_sweep = time.monotonic()
            try:
                self.storage.sweep(self.settings.retention_days)
            except Exception:
                log.exception("retention sweep failed")

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        started = time.time()
        self.store.update(job_id, status="running", started_at=started)
        self.current_job = job_id
        try:
            t0 = time.monotonic()
            backend = self.manager.get(job.model)
            t1 = time.monotonic()
            image = backend.generate(
                GenParams(job.prompt, job.width, job.height, job.steps, job.guidance, job.seed)
            )
            t2 = time.monotonic()

            done = replace(
                job, status="succeeded", started_at=started, finished_at=time.time(),
                load_seconds=round(t1 - t0, 3), generate_seconds=round(t2 - t1, 3),
            )
            meta = done.to_meta()
            # Image first, then metadata, then publish "succeeded" to the store, so a
            # client that sees "succeeded" can always fetch the image.
            self.storage.save_image(
                job_id, image,
                {k: meta[k] for k in ("id", "model", "prompt", "params", "created_at")},
            )
            self.storage.save_meta(job_id, meta)
            self.store.replace_job(done)
            log.info("job %s succeeded model=%s load=%.2fs generate=%.2fs",
                     job_id, job.model, done.load_seconds, done.generate_seconds)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            failed = replace(
                job, status="failed", started_at=started, finished_at=time.time(),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            try:
                self.storage.save_meta(job_id, failed.to_meta())
            except Exception:
                log.exception("could not persist failure metadata for %s", job_id)
            self.store.replace_job(failed)
        finally:
            self.current_job = None
