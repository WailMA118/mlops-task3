# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Runtime only: notebooks, training dependencies and test tooling are excluded.
COPY requirements/base.txt requirements/base.txt
RUN pip install --upgrade pip \
    && pip install -r requirements/base.txt

# Copy only runtime application/configuration files. The notebooks directory is
# intentionally never copied into the image.
COPY app ./app
COPY src ./src
COPY config ./config
COPY models ./models
COPY data ./data
COPY examples ./examples

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs /app/models /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
