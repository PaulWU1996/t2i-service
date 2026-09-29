# T2I Service

HTTP text-to-image service: natural-language prompt in, PNG out. Diffusers backend,
FLUX.2 [klein] 4B (Apache-2.0) by default, packaged as a GPU container.

## Run

```bash
docker compose up --build -d     # optional: cp .env.example .env to change defaults
curl localhost:8000/healthz
```

Requires Docker with the NVIDIA container toolkit. First start downloads the weights into
the `t2i-hf-cache` volume (or mount pre-provisioned weights, see below).

## API

No authentication: any client that can reach the port can call the service (see limits below).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/generations` | Submit a prompt. `202` + job; add `?wait=true[&timeout=60]` to hold until done (`200` if finished, `202` if the wait ran out). |
| GET | `/v1/generations/{id}` | Job status, parameters, timings. |
| GET | `/v1/generations/{id}/image` | The PNG (`409` if not finished, `404` if failed/expired). |
| GET | `/v1/models` | Available models and their defaults. |
| GET | `/healthz` | Liveness: loaded model, busy flag, queue depth. |

Interactive docs: `/docs`.

```bash
# simplest: one call, image comes back by URL in the response
curl -X POST 'localhost:8000/v1/generations?wait=true' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A lighthouse on a cliff at dawn, soft fog, film photograph"}'

curl localhost:8000/v1/generations/<id>/image -o out.png
```

Request fields (only `prompt` is required): `model`, `width`, `height` (multiples of 16),
`steps`, `guidance`, `seed`. Omitted fields take the model's defaults; an omitted seed is
randomised and recorded. Unknown fields (including `image`, `negative_prompt`) are rejected
with `422` rather than ignored.

Prefer async (no `wait`) for anything behind a proxy: generation plus a cold model load can
exceed proxy timeouts. `wait` is capped at 90 s by default (`T2I_MAX_WAIT_SECONDS`).

Status codes: `422` invalid request or image too large, `400` unknown
model, `429` queue full (`Retry-After: 10`), `404` unknown/expired id.

Every PNG embeds its prompt, model, seed and parameters (PNG text chunk `t2i`), and a
`<id>.json` with the same plus timings sits beside it in the outputs volume.

## Configuration (env vars, prefix `T2I_`)

| Variable | Default | Notes |
|---|---|---|
| `DEFAULT_MODEL` | `flux2-klein-4b` | |
| `PRELOAD_MODEL` | `false` | Load at startup instead of first request. |
| `CPU_OFFLOAD` | `false` | Lower VRAM, slower. |
| `MODELS_DIR` | `/data/models` | If `<MODELS_DIR>/<model-name>/` exists, load from there, not the Hub. |
| `OUTPUT_DIR` | `/data/outputs` | |
| `RETENTION_DAYS` | `30` | Old results deleted; `0` keeps everything. |
| `MAX_QUEUE_SIZE` | `32` | Waiting jobs; beyond this, `429`. |
| `MAX_SIDE` / `MAX_PIXELS` | `2048` / `2359296` | |
| `ENABLE_MOCK_MODEL` | `false` | CPU-only fake model for development. |

## Adding a model

Add a `ModelSpec` in `app/models.py` naming the diffusers pipeline class, repo, and default
steps/guidance (e.g. `Krea2Pipeline` for Krea 2 Turbo: 8 steps, guidance 0). No other code
changes. **Check the model's license first**: FLUX.2 [klein] 9B and [dev] are
non-commercial; Krea 2 is under its own community license.

## Development

```bash
pip install -r requirements-dev.txt fastapi uvicorn pydantic-settings pillow
python -m pytest                      # runs against the mock backend, no GPU needed

T2I_ENABLE_MOCK_MODEL=true T2I_DEFAULT_MODEL=mock T2I_OUTPUT_DIR=/tmp/out \
 uvicorn app.main:create_app --factory --port 8000
```

## Status and known limits

- **Tested:** the whole HTTP/queue/storage layer (27 tests, plus a live uvicorn + curl run)
  against the mock backend.
- **Not yet run:** the diffusers backend on a real GPU, and the Docker build. The pipeline
  class and call signature were checked against the diffusers 0.40.0 source, but nothing has
  loaded real weights. Pin exact dependency versions after the first successful VM run.
  `TORCH_INDEX_URL`/`TORCH_VERSION` in the Dockerfile must match the VM's driver.
- Single process, single GPU worker, in-memory job table: jobs queued or running when the
  service restarts are lost (finished results stay on disk and remain retrievable).
- No authentication, by design (the orchestrator calls it directly). Restrict access at the network level (VM security group / internal network only, not publicly exposed); if BBC standards later require auth, put a gateway in front.
- Prompts are not logged (only their length); job ids, model and timings are.
- Not implemented yet: image-to-image (the default model supports it; needs an upload field
  and pipeline `image=` argument), OpenAI-compatible `/v1/images/generations`, idle unload,
  provenance signing (C2PA), and a content-safety filter (FLUX.2 licence terms may require one).
