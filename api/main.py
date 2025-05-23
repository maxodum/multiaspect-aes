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
    """
    Request model for essay text input.
    
    Attributes:
        text (str): The essay text to be evaluated.
    """
    text: str


class TaskIDsOut(BaseModel):
    """
    Response model for asynchronous task IDs.
    
    Attributes:
        qwk_task_id (str): Task ID for the QWK aspect evaluation task.
        feedback_task_id (str): Task ID for the feedback generation task.
    """
    qwk_task_id: str
    feedback_task_id: str


@app.get("/")
def home():
    """
    Root endpoint to check the service health.
    
    Returns:
        dict: A simple health check response.
    """
    REQUEST_COUNTER.inc()
    return {"health_check": "OK"}


@app.post("/evaluate", response_model=TaskIDsOut)
async def predict(essay: TextIn, db: AsyncSession = Depends(get_async_db)):
    """
    Endpoint to start asynchronous evaluation tasks for an essay.

    This endpoint submits two Celery tasks:
        - One for aspect-based scoring (QWK).
        - One for feedback generation.
    The task IDs are saved in the database for tracking and returned to the client.

    Args:
        essay (TextIn): The essay text wrapped in a request model.
        db (AsyncSession): Database session (dependency-injected).

    Returns:
        TaskIDsOut: The task IDs for QWK and feedback tasks.
    """
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
    """
    Endpoint to check the status or retrieve results of a Celery task.

    Args:
        task_id (str): The ID of the Celery task to retrieve.
        db (AsyncSession): Database session (dependency-injected).

    Returns:
        dict: A dictionary containing the task status and result (if ready).
    """
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
