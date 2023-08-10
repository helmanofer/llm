from typing import List

from pydantic import BaseModel


class DocumentWithScore(BaseModel):
    text: str
    score: float


class QuestionAnswer(BaseModel):
    answer: str
    documents: List[DocumentWithScore]
