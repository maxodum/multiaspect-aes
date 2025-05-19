FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install poetry

COPY worker/pyproject.toml worker/poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root

COPY worker /app/worker

CMD ["celery", "-A", "worker.tasks", "worker", "--loglevel=info", "--pool=solo"]