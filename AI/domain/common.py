import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, validator


class DateTimeModelMixin(BaseModel):
    created_at: datetime.datetime = None  # type: ignore
    updated_at: datetime.datetime = None  # type: ignore

    @validator("created_at", "updated_at", pre=True)
    def default_datetime(
        cls,  # noqa: N805
        value: datetime.datetime,  # noqa: WPS110
    ) -> datetime.datetime:
        return value or datetime.datetime.now()


class IDModelMixin(BaseModel):
    id_: int = Field(0, alias="id")


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BAD_REQUEST = "BAD_REQUEST"


class CommonResponse:
    status: str
    message: str
    code: str
    image_name: str
    data: Any

    def __init__(self, status: Status = Status.SUCCESS, message: str = "Successful", code: str = "200", data=None,
                 image_name: str = ""):
        self.status = status.value
        self.message = message
        self.code = code
        self.data = data
        self.image_name = image_name

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "code": self.code,
            "data": self.data,
            "image_name": self.image_name
        }
