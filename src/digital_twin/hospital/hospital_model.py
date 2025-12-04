from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------- Basic structures ----------

@dataclass
class LOSModel:
    type: str                     # "lognormal", "gamma", "empirical_histogram", ...
    params: Dict[str, float] = field(default_factory=dict)
    source: Optional[str] = None  # for empirical: path to CSV / source


@dataclass
class Ward:
    id: str
    name: str
    type: str                     # "ED", "ICU", "WARD", "SHORT_STAY", ...
    capacity: int
    processtime_model: LOSModel


@dataclass
class PatientType:
    id: str
    description: str
    priority: int = 0             # higher value = higher priority in DES


@dataclass
class ArrivalModel:
    type: str                     # "ml_poisson", "schedule", "constant", ...
    source: Optional[str] = None  # e.g. CSV with history
    seasonal_features: List[str] = field(default_factory=list)
    mean_per_day: Optional[float] = None
    pattern: Optional[str] = None         # e.g. "Mon,Tue,Thu"
    extra: Dict[str, float] = field(default_factory=dict)  # free space (e.g. scale_factors)


@dataclass
class ArrivalStream:
    id: str
    target_ward: str              # Ward.id
    patient_type: str             # PatientType.id
    model: ArrivalModel


# ---------- Routing & policies ----------

@dataclass
class RoutingOption:
    to_ward: str                  # Ward.id
    probability: float


@dataclass
class RoutingRule:
    from_ward: str                # Ward.id
    patient_type: str             # PatientType.id
    options: List[RoutingOption]


@dataclass
class PriorityRule:
    ward: str                     # Ward.id
    patient_type: str             # PatientType.id
    priority: int


@dataclass
class OverflowRule:
    ward: str                     # Ward.id
    threshold: float
    redirect_to: str              # Ward.id


@dataclass
class AdmissionStopRule:
    ward: str                     # Ward.id
    threshold: float
    stop_types: List[str]         # list of PatientType.id


@dataclass
class Policies:
    priorities: List[PriorityRule] = field(default_factory=list)
    overflow: List[OverflowRule] = field(default_factory=list)
    admission_stop: List[AdmissionStopRule] = field(default_factory=list)



@dataclass
class HospitalMeta:
    name: str
    timezone: str = "Europe/Amsterdam"
    simulation_horizon_days: int = 180


@dataclass
class HospitalConfig:
    simulation: Dict[str, any]
    meta: HospitalMeta
    wards: Dict[str, Ward]
    patient_types: Dict[str, PatientType]
    arrival_streams: Dict[str, ArrivalStream]
    policies: Policies
    routing_rules: List[RoutingRule] = field(default_factory=list)
    
