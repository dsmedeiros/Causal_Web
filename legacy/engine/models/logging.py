import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


class PeriodicLogEntry(BaseModel):
    """Simpler schema for periodic or summary records."""

    tick: int | None = None
    label: str
    value: Any
    metadata: Dict[str, Any] | None = None


class NodeEmergencePayload(BaseModel):
    node_id: str
    origin_type: str
    parents: List[str]


class NodeEmergenceLog(PeriodicLogEntry):
    """Record the appearance of a new node."""

    label: str = "node_emergence"
    value: NodeEmergencePayload


class StructuralGrowthPayload(BaseModel):
    node_count: int
    edge_count: int
    sip_success_total: int
    csp_success_total: int
    avg_coherence: float


class StructuralGrowthLog(PeriodicLogEntry):
    """Snapshot of overall network growth metrics."""

    label: str = "structural_growth"
    value: StructuralGrowthPayload


class GenericLogEntry(BaseLogEntry):
    """Fallback model for logs without a dedicated payload class."""

    event_type: str
    payload: Any
    metadata: Dict[str, Any] | None = None
