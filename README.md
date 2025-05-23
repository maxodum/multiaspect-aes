# multiaspect-aes

**Team members**: Maxim Dumenkov (@maxodum), Polina Mashkovtseva (@pmashkovtseva)

**Project description**: Construction of Automated Essay Scoring app with several aspects as scores, generation of feedback on essay.

**Project structure:**

- notebooks: repository with the .ipynb files containing the processes of EDA and modeling experiments
- api: FastAPI app and sqlalchemy to work with PostgreSQL
- worker: Celery worker with redis as broker
- streamlit: streamlit app

**Project functionality:**

- API
    - /evaluate - post essay for evaluation and generation of feedback, receive tasks_ids
    - /result/{task_id} - get result for task (either qwk dictionary or feedback)

- Streamlit
    * Send essay, show both QWK and feedback. Used mainly for users' convenience

**Docker-compose content description:**

- api: FastAPI
- Redis: Caching and brokerage for celery
- Postgres: Database for users' essays and evaluated QWK with feedbacks
- Celery: Queue of tasks
- Prometheus & Grafana: Services monitoring
- worker: GPU worker for Celery


**Project assembling instruction:**

1. Clone the repo on your machine
```
git clone https://github.com/maxodum/multiaspect-aes
```
2. Create .env file with required fields
```
REDIS_BROKER="redis://redis:6379/0"
REDIS_BACKEND="redis://redis:6379/0"

POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=mydb

DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/mydb
DATABASE_URL_SYNC=postgresql://postgres:password@db:5432/mydb

TRANSFORMERS_CACHE=/cache/huggingface
LD_LIBRARY_PATH=/usr/local/nvidia/lib64
```
3. Build the containers (might take a while)
```
docker-compose build
```
4. Start the docker containers (first time might also take a while)
```
docker compose up
```