"""Job records and the in-memory store. Finished jobs are also persisted by Storage."""
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import time


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    model: str
    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    load_seconds: float | None = None
    generate_seconds: float | None = None
    error: str | None = None

    def to_meta(self) -> dict:
        """JSON-serialisable form; matches schemas.JobView (minus image_url)."""
        queue_s = None if self.started_at is None else round(self.started_at - self.created_at, 3)
        timings = None
        if self.status in ("succeeded", "failed"):
            timings = {
                "queue_seconds": queue_s,
                "load_seconds": self.load_seconds,
                "generate_seconds": self.generate_seconds,
            }
        return {
            "id": self.id,
            "status": self.status,
            "model": self.model,
            "prompt": self.prompt,
            "params": {
                "width": self.width,
                "height": self.height,
                "steps": self.steps,
                "guidance": self.guidance,
                "seed": self.seed,
            },
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "timings": timings,
            "error": self.error,
        }


class JobStore:
    """Thread-safe, bounded. Old entries fall out; results remain on disk."""

    def __init__(self, max_items: int = 1000):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job else None  # copy: callers never see partial updates

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in fields.items():
                    setattr(job, k, v)

    def replace_job(self, job: Job) -> None:
        with self._lock:
            if job.id in self._jobs:
                self._jobs[job.id] = job

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
