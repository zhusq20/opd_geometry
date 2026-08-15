from typing import Any

from pydantic import BaseModel

RESPOND_ACTION_NAME = "respond"


class Action(BaseModel):
    name: str
    type: str
    arguments: dict[str, Any]
