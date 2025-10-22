import numpy as np


def logistic_N(years, n_max, k, t_mid):
    """Unconstrained logistic adoption curve."""
    years = np.asarray(years)
    return n_max / (1.0 + np.exp(-k * (years - t_mid)))


def constrained_logistic_N(years, n_max, k, t_mid, max_build_rate_per_year, N0=0.0):
    """Logistic adoption with supply-chain ceiling on annual additions.
    
    Args:
        years: array of years
        n_max: maximum cumulative reactors
        k: logistic steepness parameter
        t_mid: year when 50% of n_max is reached
        max_build_rate_per_year: maximum annual reactor additions
        N0: starting cumulative count
    
    Returns:
        Array of cumulative reactors N(t) with supply-chain constraint
    """
    years = np.asarray(years)
    
    # Unconstrained logistic curve
    N_logistic = logistic_N(years, n_max, k, t_mid)
    
    # Apply supply-chain ceiling by constraining annual additions
    dN_raw = np.diff(np.r_[N0, N_logistic])
    dN_constrained = np.minimum(dN_raw, max_build_rate_per_year)
    
    # Re-integrate to get constrained cumulative trajectory
    N_constrained = np.cumsum(dN_constrained) + N0
    
    # Ensure we don't exceed n_max
    N_final = np.minimum(N_constrained, n_max)
    
    return N_final


def bottom_up_N(years, total_customers, build_years, max_build_rate_per_year, N0=0.0):
    """Deterministic bottom-up ramp: add customers evenly over `build_years`,
    respecting a per-year build ceiling. N0 is the starting installed base.
    """
    years = np.asarray(years)
    # Target uniform additions per year
    per_year_target = total_customers / max(1, build_years)
    per_year = np.minimum(per_year_target, max_build_rate_per_year)
    adds = np.zeros_like(years, dtype=float)
    y0 = years[0]
    horizon = min(build_years, len(years))
    adds[:horizon] = per_year
    N = np.cumsum(adds) + N0
    return np.minimum(N, total_customers + N0)


def constrained_logistic_with_ceiling_ramp(
    years,
    n_max: float,
    k: float,
    t_mid: float,
    ceiling_start_per_year: float,
    ceiling_growth: float,
    ceiling_max_per_year: float,
    N0: float = 0.0,
):
    """Logistic adoption constrained by a time-varying (ramping) build ceiling.

    The annual build ceiling ramps up exponentially from a small initial value,
    capped at ceiling_max_per_year. This avoids early spikes and prevents
    mid-horizon slowdowns by allowing the supply chain to expand over time.
    """
    years = np.asarray(years)
    N_logistic = logistic_N(years, n_max, k, t_mid)

    # Compute time-varying ceiling per year
    t0 = years[0]
    growth_years = (years - t0)
    ceiling_series = ceiling_start_per_year * np.exp(ceiling_growth * growth_years)
    ceiling_series = np.minimum(ceiling_series, ceiling_max_per_year)

    # Constrain annual increments by the per-year ceiling
    dN_raw = np.diff(np.r_[N0, N_logistic])
    dN_constrained = np.minimum(dN_raw, ceiling_series)
    N_constrained = np.cumsum(dN_constrained) + N0
    return np.minimum(N_constrained, n_max)


def exponential_additions_with_ceiling(
    years,
    start_additions_per_year: float,
    growth: float,
    ceiling_max_per_year: float,
    N0: float = 0.0,
):
    """Pure exponential annual additions capped by a ceiling; integrates to N(t).

    Ensures additions never decrease on the modeled horizon, reflecting growing
    human energy demand and expanding supply chains.
    """
    years = np.asarray(years)
    t0 = years[0]
    growth_years = (years - t0)
    adds = start_additions_per_year * np.exp(growth * growth_years)
    adds = np.minimum(adds, ceiling_max_per_year)
    N = np.cumsum(adds) + N0
    return N


