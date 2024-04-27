# https://gist.github.com/marcelo-clarifai/3f616a9e7bbb75d062ad79fb959d2f16

##############
# Base Image #
##############
FROM python:3.10.14-slim-bookworm as builder

RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \  
    PYSETUP_PATH="/opt/pysetup"
    
RUN curl -sSL https://install.python-poetry.org | python3
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root


#####################
# Development Image #
#####################
FROM builder as development
ENV FASTAPI_ENV=development
COPY --from=builder $POETRY_HOME $POETRY_HOME
COPY --from=builder $PYSETUP_PATH $PYSETUP_PATH
WORKDIR $PYSETUP_PATH
RUN poetry install --no-interaction 
WORKDIR /app
COPY . .
CMD ["uvicorn", "symbiont.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#####################
# Production Image  #
#####################
FROM builder as prodcution
ENV FASTAPI_ENV=prodcution
COPY --from=builder $POETRY_HOME $POETRY_HOME
COPY --from=builder $PYSETUP_PATH $PYSETUP_PATH
WORKDIR $PYSETUP_PATH
RUN poetry install --no-interaction 
WORKDIR /app
COPY . .
EXPOSE 8080
CMD ["uvicorn", "symbiont.main:app", "--host", "0.0.0.0", "--port", "8080"]