FROM python:3.12-slim

# torch must match the VM's NVIDIA driver. cu128 needs a recent driver (Blackwell GPUs need >= cu128).
# Override at build time, e.g.:  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.9.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    T2I_OUTPUT_DIR=/data/outputs \
    T2I_MODELS_DIR=/data/models

RUN useradd --create-home --uid 10001 app
WORKDIR /srv

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
RUN mkdir -p /data/outputs /data/models /data/hf && chown -R app:app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

# One process only: the model lives in this process's GPU memory. Do not add --workers.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
