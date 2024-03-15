# Use an official Python runtime as a parent image
FROM python:3.9 as builder

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# Now copy all project
COPY . /app

# Production image
FROM python:3.9-slim as production

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Run the app
CMD ["uvicorn", "symbiont.src.main:app", "--host", "0.0.0.0", "--port", "8000"]