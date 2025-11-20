# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.13-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (production only, no dev/notebooks)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

# Copy Python environment from builder
COPY --from=builder --chown=apiuser:apiuser /app/.venv /app/.venv

# Copy application code
COPY --chown=apiuser:apiuser ./api ./api
COPY --chown=apiuser:apiuser ./src ./src

# Copy models directory (will be empty initially, can be mounted as volume)
COPY --chown=apiuser:apiuser ./models ./models

# Switch to non-root user
USER apiuser

# Add .venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

# Run application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"]
