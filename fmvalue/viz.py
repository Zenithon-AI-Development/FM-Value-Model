import numpy as np
import pandas as pd


def ribbons(df: pd.DataFrame, key: str, qs=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    grouped = df.groupby("year")[key]
    q_df = grouped.quantile(qs).unstack()
    q_df.columns = [f"q{int(q*100)}" for q in qs]
    q_df = q_df.reset_index()
    return q_df


def waterfall_components(run_fn, cfg, seed_base: int, year: int):
    """Compute per-channel LCOE contributions for waterfall analysis.
    
    Isolates each FM channel's impact by running with only that channel enabled.
    
    Args:
        run_fn: Monte Carlo run function (should accept channel parameter)
        cfg: configuration object
        seed_base: random seed for reproducibility
        year: target year for evaluation
    
    Returns:
        Tuple of (component_list, total_delta)
        where component_list = [(channel_name, delta_lcoe), ...]
    """
    # Run baseline (no FM)
    base = run_fn(cfg, seed_base, with_fm=False)
    years_arr = base["years"]
    idx = int(year - years_arr[0]) if year >= years_arr[0] else 0
    base_lcoe = float(base["lcoe"][idx])
    
    # Run full FM scenario
    full = run_fn(cfg, seed_base + 1, with_fm=True)
    full_lcoe = float(full["lcoe"][idx])
    total_delta = full_lcoe - base_lcoe
    
    # For now, return stub values (proper implementation requires channel toggles in run_fn)
    # TODO: Implement per-channel runs when run_fn supports channel parameter
    components = [
        ("simulation", 0.0),
        ("experiments", 0.0),
        ("control", 0.0),
        ("sharing", 0.0)
    ]
    
    return components, total_delta


def tornado_sensitivity(eval_fn, baseline_inputs: dict, variations: dict, year: int):
    results = []
    for name, (low, high) in variations.items():
        l = eval_fn({**baseline_inputs, name: low}, year)
        h = eval_fn({**baseline_inputs, name: high}, year)
        results.append((name, l, h))
    df = pd.DataFrame(results, columns=["name", "low", "high"]) 
    df["range"] = (df[["low", "high"]].max(axis=1) - df[["low", "high"]].min(axis=1)).abs()
    return df.sort_values("range", ascending=False)


