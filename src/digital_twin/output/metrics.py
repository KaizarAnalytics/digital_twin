import numpy as np
def risk_summary(max_occ, overflow_days, beds:int):
    return {
        "P(max>100%)": float(np.mean(max_occ>beds)),
        "P(max>95%)": float(np.mean(max_occ>0.95*beds)),
        "mean_overflow_days": float(np.mean(overflow_days)),
        "p95_max_occ": float(np.percentile(max_occ,95)),
        "median_max_occ": float(np.median(max_occ)),
    }
