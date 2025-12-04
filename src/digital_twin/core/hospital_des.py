from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import simpy
import copy

from digital_twin.hospital.hospital_model import HospitalConfig
from digital_twin.output.flow_graph import build_flow_graph_from_config, FlowGraph
from digital_twin.core.processtime_factory import make_processtime_sampler_for_ward


@dataclass
class WardRunSummary:
    ward_id: str
    beds: int
    rep: int
    mean_wait: float
    p95_wait: float
    max_wait: float
    mean_occupancy: float
    p95_occupancy: float
    days_over_95pct: float


class HospitalDES:
    """
    Multi-ward DES:
      - one SimPy Environment
      - multiple ward resources
      - patients flowing through via FlowGraph
    """

    def __init__(
        self,
        env: simpy.Environment,
        cfg: HospitalConfig,
        processtime_samplers: Dict[str, callable],
        flow_graph: FlowGraph,
        warmup_days: int = 14,
        processtime_factors: Dict[str, float] | None = None,
        min_processtime: float = 0.1,
    ):
        self.env = env
        self.cfg = cfg
        self.flow_graph = flow_graph
        self.processtime_samplers = processtime_samplers
        self.warmup_days = warmup_days

        self.processtime_factors: Dict[str, float] = processtime_factors or {}
        self.min_processtime = float(min_processtime)
        

        self.resources: Dict[str, simpy.Resource] = {
            ward_id: simpy.Resource(env, capacity=ward.capacity)
            for ward_id, ward in cfg.wards.items()
        }

        self.capacities: Dict[str, int] = {
            ward_id: ward.capacity for ward_id, ward in cfg.wards.items()
        }

        # Stats
        self.waits: Dict[str, List[float]] = {ward_id: [] for ward_id in cfg.wards}
        self.occ_samples: List[tuple] = []

    def sample_processtime(self, ward_id: str) -> float:
        """Base LOS from sampler × scenario factor, with minimum."""
        base_processtime = float(self.processtime_samplers[ward_id]())
        factor = float(self.processtime_factors.get(ward_id, 1.0))
        processtime = base_processtime * factor
        if processtime < self.min_processtime:
            processtime = self.min_processtime
        return processtime



    # ----------------- process ----------------- #

    def patient_flow(
        self,
        patient_id: int,
        start_ward: str,
        arrival_time: float,
        rng: np.random.Generator,
    ):
        """
        One patient flow process:
          - arrives at start_ward at arrival_time
          - passes through wards according to FlowGraph
          - leaves system at "EXIT"
        """
        env = self.env
        if env.now < arrival_time:
            yield env.timeout(arrival_time - env.now)

        ward_id = start_ward

        while ward_id != "EXIT":
            if ward_id not in self.resources:
                # safety: unknown node -> exit
                break

            beds = self.resources[ward_id]

            arrival = env.now
            with beds.request() as req:
                yield req
                wait = env.now - arrival
                self.waits[ward_id].append(wait)

                # LOS-sample
                processtime = self.sample_processtime(ward_id)
                
                yield env.timeout(processtime)

            ward_id = self.flow_graph.next_ward(ward_id, rng)

    def arrival_process(
        self,
        entry_ward: str,
        mu_full: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        Daily arrivals at entry_ward according to mu_full (including warmup).
        Time unit = days.
        """
        env = self.env
        current_day = 0.0
        patient_id = 0

        for mu in mu_full:
            n = rng.poisson(mu)
            for _ in range(n):
                offset = rng.uniform(0.0, 1.0)
                arrival_time = current_day + offset
                env.process(
                    self.patient_flow(
                        patient_id=patient_id,
                        start_ward=entry_ward,
                        arrival_time=arrival_time,
                        rng=rng,
                    )
                )
                patient_id += 1
            current_day += 1.0
            yield env.timeout(1.0)

    def monitor_occupancy(self, sim_time: float, sample_step: float = 1.0):
        """
        Log occupancy in all wards every sample_step.
        """
        env = self.env
        while env.now < sim_time:
            for ward_id, res in self.resources.items():
                self.occ_samples.append((env.now, ward_id, res.count))
            yield env.timeout(sample_step)

    # ----------------- metrics ----------------- #

    def to_dataframe(self, rep: int) -> pd.DataFrame:
        """
        Aggregate to ward-level metrics for this replication.
        """
        if self.occ_samples:
            df_occ = pd.DataFrame(
                self.occ_samples,
                columns=["time", "ward_id", "occupancy"],
            )
        else:
            df_occ = pd.DataFrame(columns=["time", "ward_id", "occupancy"])

        summaries: List[WardRunSummary] = []

        for ward_id, cap in self.capacities.items():
            waits = np.array(self.waits.get(ward_id, []), dtype=float)
            if waits.size == 0:
                waits = np.array([0.0])

            occ_w = df_occ[df_occ["ward_id"] == ward_id]
            if not occ_w.empty:
                occ_vals = occ_w.loc[occ_w["time"] >= self.warmup_days, "occupancy"]
                if occ_vals.empty:
                    occ_vals = occ_w["occupancy"]
                occ_arr = occ_vals.to_numpy(dtype=float)
            else:
                occ_arr = np.array([0.0])

            mean_occ = float(occ_arr.mean())
            p95_occ = float(np.percentile(occ_arr, 95)) if occ_arr.size > 0 else 0.0
            days_over_95 = float((occ_arr > 0.95 * cap).sum())

            summary = WardRunSummary(
                ward_id=ward_id,
                beds=int(cap),
                rep=rep,
                mean_wait=float(waits.mean()),
                p95_wait=float(np.percentile(waits, 95)),
                max_wait=float(waits.max()),
                mean_occupancy=mean_occ,
                p95_occupancy=p95_occ,
                days_over_95pct=days_over_95,
            )
            summaries.append(summary)

        return pd.DataFrame([s.__dict__ for s in summaries])


def _wrap_processtime_sampler(base_sampler, rng):
    """
    Make a 0-argument sampler:
      - tries different call signatures:
        (rng, 1), (1, rng), (rng), (1)
      - takes the first that does not raise a TypeError
      - converts the result to a float (scalar)
    """
    def one_sample():
        last_err = None
        # Try a few common patterns
        for fn in (
            lambda: base_sampler(rng, 1),
            lambda: base_sampler(1, rng),
            lambda: base_sampler(rng),
            lambda: base_sampler(1),
        ):
            try:
                vals = fn()
                break
            except TypeError as e:
                last_err = e
        else:
            # If all fail, the sampler just has a silly signature
            raise TypeError(
                f"Could not call LOS sampler with any of the tried signatures: {last_err}"
            )

        # scalar or array → always return one float
        try:
            return float(vals[0])
        except (TypeError, IndexError, KeyError):
            return float(vals)

    return one_sample



def run_hospital_des(
    cfg: HospitalConfig,
    entry_ward: str,
    mu_series: pd.Series,
    patients_df: pd.DataFrame,
    warmup_days: int = 14,
    n_rep: int = 200,
    seed: int | None = None,
    min_processtime: float = 0.5,
    capacity_overrides: Dict[str, int] | None = None,
    processtime_factors: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Multi-replication hospital DES.

    Parameters
    ----------
    cfg : HospitalConfig
        Hospital configuration (wards, routing, etc.).
    entry_ward : str
        Ward-id where patients enter (e.g., "SEH" or "WARD_A").
    mu_series : pd.Series
        Daily arrival intensity at entry_ward.
    patients_df : pd.DataFrame
        Raw patient data, used by processtime_factory (type 'from_data').
    warmup_days : int
        Extra days to bring the system to equilibrium.
    n_rep : int
        Number of replications.
    seed : int
        Base seed for random generator.
    min_processtime : float
        Minimum LOS in days.
    capacity_overrides : dict[ward_id, capacity]
        Optional: overrides capacity per ward (for scenarios).
    processtime_factors : dict[ward_id, factor]
        Optional: multiplies LOS per ward (for scenarios).

    Returns
    -------
    pd.DataFrame
        Metrics per ward per replication.
    """
    if entry_ward not in cfg.wards:
        raise ValueError(f"Entry ward '{entry_ward}' not in cfg.wards.")

    capacity_overrides = capacity_overrides or {}
    processtime_factors = processtime_factors or {}

    # --- arrivals / horizon ---
    mu_series = mu_series.sort_index()
    mu_vals = mu_series.values.astype(float)
    horizon_days = len(mu_vals)

    # --- build flow-graph from base config ---
    flow_graph = build_flow_graph_from_config(cfg)

    all_dfs: List[pd.DataFrame] = []

    for rep in range(n_rep):
        cfg_rep = copy.deepcopy(cfg)

        # apply capacity overrides
        for wid, cap in capacity_overrides.items():
            if wid in cfg_rep.wards:
                cfg_rep.wards[wid].capacity = int(cap)
        rndseed = seed + rep if seed is not None else None
        rng = np.random.default_rng(rndseed)
        env = simpy.Environment()

        # LOS-samplers per ward, for this replication
        processtime_samplers: Dict[str, callable] = {}
        for ward_id in cfg_rep.wards.keys():
            base_sampler = make_processtime_sampler_for_ward(
                cfg_rep,
                ward_id=ward_id,
                patients_df=patients_df,
                min_processtime=min_processtime,
            )

            # make a 0-arg sampler that figures out how it wants to be called
            processtime_samplers[ward_id] = _wrap_processtime_sampler(base_sampler, rng)




        des = HospitalDES(
            env=env,
            cfg=cfg_rep,
            processtime_samplers=processtime_samplers,
            flow_graph=flow_graph,
            warmup_days=warmup_days,
            processtime_factors=processtime_factors,
            min_processtime=min_processtime,
        )

        mean_mu = float(mu_vals.mean()) if horizon_days > 0 else 0.0
        mu_full = np.concatenate([np.full(warmup_days, mean_mu), mu_vals])
        sim_time = warmup_days + horizon_days + 30

        env.process(des.arrival_process(entry_ward, mu_full, rng))
        env.process(des.monitor_occupancy(sim_time))

        env.run(until=sim_time)

        df_rep = des.to_dataframe(rep=rep)
        all_dfs.append(df_rep)

    return pd.concat(all_dfs, ignore_index=True)
