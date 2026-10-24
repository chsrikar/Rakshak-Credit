from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monthly_income: float = Field(ge=0, le=10_000_000)
    essential_expenses: float = Field(ge=0, le=10_000_000)
    existing_monthly_debt: float = Field(ge=0, le=10_000_000)
    monthly_work_costs: float = Field(default=0, ge=0, le=10_000_000)
    safety_buffer: float = Field(default=2000, ge=0, le=10_000_000)
    repayment_share: float = Field(default=0.35, gt=0, le=1)
    monthly_interest_rate: float = Field(default=0.02, ge=0, le=1)
    tenure_months: int = Field(default=12, ge=1, le=120)

from typing import Any

class ScoreRequest(BaseModel):
    features: dict[str, Any]

class NeuralScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]
