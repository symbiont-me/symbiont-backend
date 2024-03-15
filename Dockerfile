FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* /app/


# poetry already creates a virtualenv, so we need to disable it
RUN poetry config virtualenvs.create false 
RUN poetry install --no-interaction

CMD ["uvicorn", "symbiont.src.main:app", "--host", "0.0.0.0", "--port", "8000"]