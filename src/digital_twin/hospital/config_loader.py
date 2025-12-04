from __future__ import annotations

import pathlib
from typing import Dict, Any

import yaml

from digital_twin.hospital.hospital_model import (
    HospitalMeta,
    HospitalConfig,
    Ward,
    LOSModel,
    Policies,
)


def _load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Root of YAML {path} is not a mapping.")
    return data


def build_hospital_config_from_dict(raw: dict) -> HospitalConfig:
    # ---- meta ----
    hospital_meta_raw = raw.get("hospital", {})
    meta = HospitalMeta(
        name=hospital_meta_raw.get("name", "Unknown hospital"),
        timezone=hospital_meta_raw.get("timezone", "Europe/Amsterdam"),
        simulation_horizon_days=int(
            hospital_meta_raw.get("simulation_horizon_days", 180)
        ),
    )

    # ---- wards ----
    wards_raw: list[dict] = raw.get("wards", [])
    wards: dict[str, Ward] = {}
    for w in wards_raw:
        processtime_raw = w.get("los_model", w.get("processtime_model", {}))
        processtime_model = LOSModel(
            type=str(processtime_raw.get("type", "from_data")),
            params=processtime_raw.get("params", {}) or {},
            source=processtime_raw.get("source"),
        )
        ward = Ward(
            id=str(w["id"]),
            name=str(w.get("name", w["id"])),
            type=str(w.get("type", "WARD")),
            capacity=int(w.get("capacity", 0)),
            processtime_model=processtime_model,
        )
        wards[ward.id] = ward


    routes_raw: list[dict] = raw.get("routes", []) or []
    routes_simple = []
    for r in routes_raw:
        frm = str(r.get("from"))
        for dest in r.get("to", []) or []:
            routes_simple.append(
                {
                    "from": frm,
                    "to": str(dest.get("id")),
                    "p": float(dest.get("p", 1.0)),
                }
            )

    empty_policies = Policies(
        priorities=[],
        overflow=[],
        admission_stop=[],
    )

    return HospitalConfig(
        simulation=raw.get("simulation", {}),
        meta=meta,
        wards=wards,
        patient_types={},
        arrival_streams={},
        policies=empty_policies,
        routing_rules=routes_simple,
    )


def load_hospital_config(path: str | pathlib.Path) -> HospitalConfig:
    raw = _load_yaml(path)
    return build_hospital_config_from_dict(raw)

