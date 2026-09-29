"""On-disk results: <output_dir>/<id>.png and <output_dir>/<id>.json (atomic writes)."""
import json
import logging
import os
import re
import time
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

log = logging.getLogger("t2i.storage")

ID_RE = re.compile(r"^[0-9a-f]{32}$")


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def image_path(self, job_id: str) -> Path:
        assert ID_RE.match(job_id)
        return self.root / f"{job_id}.png"

    def meta_path(self, job_id: str) -> Path:
        assert ID_RE.match(job_id)
        return self.root / f"{job_id}.json"

    def save_image(self, job_id: str, image: Image.Image, png_meta: dict) -> None:
        """Save as PNG with the generation parameters embedded (reproducibility / provenance)."""
        info = PngInfo()
        info.add_text("t2i", json.dumps(png_meta))  # ascii-escaped, safe for tEXt
        path = self.image_path(job_id)
        tmp = path.with_name(path.name + ".tmp")
        image.save(tmp, format="PNG", pnginfo=info)
        os.replace(tmp, path)

    def save_meta(self, job_id: str, meta: dict) -> None:
        path = self.meta_path(job_id)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, path)

    def load_meta(self, job_id: str) -> dict | None:
        try:
            return json.loads(self.meta_path(job_id).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def sweep(self, retention_days: int) -> int:
        """Delete result files older than retention_days. Returns files removed."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for p in self.root.iterdir():
            if p.suffix in {".png", ".json", ".tmp"} and p.stat().st_mtime < cutoff:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    log.warning("could not remove %s", p)
        if removed:
            log.info("retention sweep removed %d files", removed)
        return removed
