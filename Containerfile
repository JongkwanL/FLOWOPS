# syntax=docker/dockerfile:1.7
# Multi-stage build optimized for ML workloads with BuildKit cache mounts

# Builder stage - compile dependencies
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Copy requirements
WORKDIR /build
COPY requirements.txt .

# Build wheels with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt

# Runtime stage - minimal image with CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

# Install Python runtime and essential packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser -m -s /bin/bash mluser

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install dependencies from wheels (no compilation needed)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links /wheels /wheels/*.whl && \
    rm -rf /wheels

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=mluser:mluser pipelines/ ./pipelines/
COPY --chown=mluser:mluser mlflow/ ./mlflow/
COPY --chown=mluser:mluser dvc/ ./dvc/
COPY --chown=mluser:mluser params.yaml ./
COPY --chown=mluser:mluser dvc.yaml ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/metrics /app/reports /app/artifacts && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow:5000} \
    PATH="/home/mluser/.local/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mlflow; print('healthy')" || exit 1

# Default command
ENTRYPOINT ["python"]
CMD ["pipelines/training/train.py"]

# Serving stage - for model serving
FROM runtime AS serving

# Install serving dependencies
USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi uvicorn[standard] prometheus-client

# Copy serving code
COPY --chown=mluser:mluser pipelines/deployment/serve.py ./

USER mluser

# Expose ports for API and metrics
EXPOSE 8080 8000

# Serving command
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

# Development stage - includes Jupyter and dev tools
FROM runtime AS development

USER root

# Install development tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    jupyter \
    jupyterlab \
    ipython \
    ipdb \
    pytest-watch \
    pre-commit

# Create Jupyter config
USER mluser
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]