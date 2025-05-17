from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult


celery = Celery(
  "inference",
  broker="redis://localhost:6379/0",
  backend="redis://localhost:6379/0"
)


app = FastAPI()


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
    return {'health_check': 'OK'}


@app.post("/evaluate", response_model=TaskIDsOut)
def predict(essay:TextIn):
    qwk_async = celery.send_task("evaluate_qwk", args=[essay.text])
    feedback_async = celery.send_task("evaluate_feedback",  args=[essay.text])
    return TaskIDsOut(qwk_task_id=qwk_async.id,
                         feedback_task_id=feedback_async.id)


@app.get("/result/{task_id}")
def get_result(task_id: str):
    result = AsyncResult(task_id, app=celery)
    if result.ready():
        return {"status": result.status, "result": result.result}
    else:
        return {"status": result.status}


#@app.post("/feedback", response_model=FeedbackOut)
#def predict(essay:TextIn):
#    feedback = give_feedback(essay.text)
#    return feedback