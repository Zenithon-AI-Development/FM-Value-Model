import numpy as np


def ops_learning_factors(
    years,
    t0,
    tau_cf=25.0,
    tau_om=30.0,
    cf_fm_boost=0.0,
    om_fm_boost=0.0,
    max_cf_gain=0.50,
    max_om_reduction=0.50,
):
    """Operational learning factors: CF and O&M improve over time via saturation curves.
    
    This captures the benefit of building experience on operations independent of CAPEX learning.
    Includes both efficiency improvements (higher CF/power output) and cost reductions (lower O&M).
    FM accelerates this learning curve.
    
    Args:
        years: array of years
        t0: reference year (start of deployment)
        tau_cf: time constant for CF maturation (years to reach ~63% of improvement)
        tau_om: time constant for O&M reduction (years to reach ~63% of reduction)
        cf_fm_boost: FM multiplier for CF improvement speed (e.g., 0.3 = 30% faster maturation)
        om_fm_boost: FM multiplier for O&M reduction speed
    
    Returns:
        Tuple of (cf_multiplier, om_multiplier) by year
        - cf_multiplier: 1.0 → 1.X (accounts for both CF and power output improvements)
        - om_multiplier: 1.0 → 0.X (O&M cost reduction)
    """
    years = np.asarray(years)
    t_elapsed = years - t0
    
    # Saturation curves: factor = 1.0 + improvement * (1 - exp(-t/tau))
    # For CF/power output: improves by up to max_cf_gain (default 50%) over long horizon
    # Physics allows marginal but compounding gains; over 50 years could approach 100% in FM case
    cf_improvement = max_cf_gain * (1.0 - np.exp(-t_elapsed / (tau_cf * (1.0 - cf_fm_boost + 1e-9))))
    cf_multiplier = 1.0 + cf_improvement
    
    # For O&M: reduces by up to max_om_reduction (default 50%) over long horizon
    om_reduction = max_om_reduction * (1.0 - np.exp(-t_elapsed / (tau_om * (1.0 - om_fm_boost + 1e-9))))
    om_multiplier = 1.0 - om_reduction
    
    # Apply bounds (allow larger long-term improvement)
    cf_multiplier = np.clip(cf_multiplier, 1.0, 1.80)  # Allow up to 80% improvement if horizon is long
    om_multiplier = np.clip(om_multiplier, 0.40, 1.0)  # Allow up to 60% reduction
    
    return cf_multiplier, om_multiplier


def power_output_multiplier(
    years,
    t0,
    per_decade: float = 0.01,
    fm_boost: float = 0.1,
    cap: float = 2.0,
):
    """Separate learning curve for effective net power output (nameplate/effective MW).

    Uses a simple exponential improvement: ~per_decade fractional gain every 10 years,
    optionally accelerated by FM (fm_boost), and capped to avoid unrealistic growth.

    multiplier(t) = exp(r_eff * max(0, t - t0)),
    where r_eff = ln(1 + per_decade_eff) / 10 and per_decade_eff = per_decade * (1 + fm_boost).

    Args:
        years: array of years
        t0: reference year (start of deployment)
        per_decade: baseline fractional improvement per decade (e.g., 0.10 = 10%/decade)
        fm_boost: fractional boost to per_decade (e.g., 0.2 → 12%/decade)
        cap: upper cap on multiplier to prevent runaway growth

    Returns:
        Numpy array of multipliers ≥ 1.0
    """
    import numpy as np

    years = np.asarray(years)
    t_elapsed = np.maximum(0.0, years - t0)
    per_decade_eff = max(0.0, per_decade) * (1.0 + max(0.0, fm_boost))
    # Convert decade improvement to continuous yearly rate
    r_eff = np.log(1.0 + per_decade_eff) / 10.0
    mult = np.exp(r_eff * t_elapsed)
    return np.clip(mult, 1.0, cap)

