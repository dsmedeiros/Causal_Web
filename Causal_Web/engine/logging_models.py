import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


def new_log_id() -> str:
    """Return a unique identifier for a log entry."""
    return f"log_{uuid.uuid4()}"


class BaseLogEntry(BaseModel):
    """Common metadata for all log entries."""

    log_id: str = Field(default_factory=new_log_id)
    tick: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None


class NodeEmergencePayload(BaseModel):
    node_id: str
    origin_type: str
    parents: List[str]


class NodeEmergenceLog(BaseLogEntry):
    event_type: str = "NodeEmerged"
    payload: NodeEmergencePayload


class StructuralGrowthPayload(BaseModel):
    node_count: int
    edge_count: int
    sip_success_total: int
    csp_success_total: int
    avg_coherence: float


class StructuralGrowthLog(BaseLogEntry):
    event_type: str = "StructuralGrowthSnapshot"
    payload: StructuralGrowthPayload
