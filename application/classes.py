from pydantic import BaseModel
from typing import List


class Item(BaseModel):
    id: str
    name: str
    props: List[str]


class Request(BaseModel):
    items: List[Item]


class SingleResponse(BaseModel):
    id: str
    reference_id: str


class Response(BaseModel):
    items: List[SingleResponse]
