from fastapi import Header
from pydantic import BaseModel


class Principal(BaseModel):
    user_id: str
    group_ids: list[str]


async def get_principal(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_group_ids: str | None = Header(default=None, alias="X-Group-Ids"),
) -> Principal:
    groups = [item.strip() for item in (x_group_ids or "local").split(",") if item.strip()]
    return Principal(user_id=x_user_id or "local-user", group_ids=groups)

