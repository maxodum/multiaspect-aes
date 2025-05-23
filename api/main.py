import os

from celery import Celery
from celery.result import AsyncResult
from db.models import FeedbackTask, QwkTask
from db.session import get_async_db
from fastapi import Depends, FastAPI
from prometheus_client import Counter, Gauge, start_http_server
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

celery = Celery(
    "inference", broker=os.getenv("REDIS_BROKER"), backend=os.getenv("REDIS_BACKEND")
)


app = FastAPI()

start_http_server(8001)

REQUEST_COUNTER = Counter("http_requests_total", "Total HTTP Requests")


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    qwk: dict
    feedback: str


class TaskIDsOut(BaseModel):
    qwk_task_id: str
    feedback_task_id: str


@app.get("/")
def home():
    REQUEST_COUNTER.inc()
    return {"health_check": "OK"}


@app.post("/evaluate", response_model=TaskIDsOut)
async def predict(essay: TextIn, db: AsyncSession = Depends(get_async_db)):
    REQUEST_COUNTER.inc()
    qwk_async = celery.send_task("evaluate_qwk", args=[essay.text])
    feedback_async = celery.send_task("evaluate_feedback", args=[essay.text])

    db.add_all(
        [
            QwkTask(task_id=qwk_async.id, text=essay.text),
            FeedbackTask(task_id=feedback_async.id, text=essay.text),
        ]
    )
    await db.commit()

    return TaskIDsOut(qwk_task_id=qwk_async.id, feedback_task_id=feedback_async.id)


@app.get("/result/{task_id}")
async def get_result(task_id: str, db: AsyncSession = Depends(get_async_db)):
    REQUEST_COUNTER.inc()
    result = AsyncResult(task_id, app=celery)
    if result.ready():
        if isinstance(result.result, dict):
            record = await db.get(QwkTask, task_id)
        else:
            record = await db.get(FeedbackTask, task_id)
        record.result = result.result
        await db.commit()
        return {"status": result.status, "result": result.result}
    else:
        return {"status": result.status}
