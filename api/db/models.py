from sqlalchemy import (
    Column,
    String,
    JSON,
    Text,
)
from .base import Base

class QwkTask(Base):
    __tablename__ = "qwk_tasks"
    task_id = Column(String, primary_key=True, index=True)
    text = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)


class FeedbackTask(Base):
    __tablename__ = "feedback_tasks"
    task_id = Column(String, primary_key=True, index=True)
    text = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
