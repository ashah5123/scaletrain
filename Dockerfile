FROM python:3.11-slim

# System libraries required by PyTorch on Linux (not bundled in manylinux wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry into the system Python.
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Disable virtualenv creation so packages install directly into the system Python.
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Copy the dependency manifest first.
# This layer is cached until pyproject.toml changes, keeping rebuilds fast.
COPY pyproject.toml ./

# Install runtime dependencies only (no project package yet).
RUN poetry install --no-root --only main

# Copy source code, then install the project package and register CLI entry points.
COPY src/ ./src/
RUN poetry install --only main

# Ensure the src layout is on the path for any direct Python invocations.
ENV PYTHONPATH=/app/src

CMD ["scaletrain"]
