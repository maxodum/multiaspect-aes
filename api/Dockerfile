FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --with main --no-interaction --no-root
    
COPY . .

COPY wait-for-db-and-migrate.sh /wait-for-db-and-migrate.sh
RUN chmod +x /wait-for-db-and-migrate.sh

ENTRYPOINT ["/wait-for-db-and-migrate.sh"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]