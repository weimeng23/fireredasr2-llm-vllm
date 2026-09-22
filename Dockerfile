# syntax=docker/dockerfile:1
ARG VLLM_VERSION=0.29.0
FROM vllm/vllm-openai:v${VLLM_VERSION}
# Pull the global ARG into this build stage.
ARG VLLM_VERSION
# Ubuntu 24.04 uses DEB822 sources; the legacy /etc/apt/sources.list is absent.
# The build installs no apt packages, so this only helps anyone adding tools
# inside a running container. Override with the upstream URL outside China.
ARG APT_MIRROR=http://mirrors.cloud.tencent.com/ubuntu/
USER root
RUN sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/ubuntu.sources
# The base image's system Python 3.12 stays the only interpreter: without this,
# an interpreter lookup that misses makes uv silently download a managed one.
ENV UV_PYTHON_DOWNLOADS=never
# UV_CACHE_DIR is /opt/uv/cache in the base image; mounting it keeps downloaded
# wheels out of the image layers, which UV_LINK_MODE=copy would otherwise duplicate.
# No `uv pip check --system` here: the base image deliberately overrides torch's
# nvidia-nccl-cu13==2.29.7 pin with 2.30.7 (DeepEPv2 needs >= 2.30.4), so a
# system-wide check always reports that one incompatibility.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system "vllm[audio]==${VLLM_VERSION}" "kaldi-native-fbank>=1.18.7"
WORKDIR /opt/fireredasr2-vllm
COPY requirements.txt ./requirements.txt
# Keep gateway pins independent of vLLM's FastAPI / Starlette dependency set.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv venv /opt/gateway-venv --python "$(command -v python3)" \
    && uv pip install --python /opt/gateway-venv/bin/python -r requirements.txt \
    && uv pip check --python /opt/gateway-venv/bin/python
COPY scripts ./scripts
COPY gateway ./gateway
COPY .env.example ./.env.example
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONUNBUFFERED=1 SERVICE_MODE=vllm
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=4 \
    CMD ["python3", "/opt/fireredasr2-vllm/scripts/healthcheck.py"]
# Clear the vLLM base image entrypoint so commands after the image replace CMD.
ENTRYPOINT []
CMD ["python3", "/opt/fireredasr2-vllm/scripts/entrypoint.py"]
