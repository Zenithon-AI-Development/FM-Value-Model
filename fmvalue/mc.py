import numpy as np
from .sampling import sampled_config
from .fm_levers import map_fm_to_knobs
from .guards import (
    assert_no_double_count,
    assert_capex_monotone_when_g0,
    assert_cf_bounds,
)
from .experiments import time_to_gate_days, fm_adjust_experiment_inputs
from .schedule import draw_schedule_months
from .adoption import (
    constrained_logistic_N,
    bottom_up_N,
    constrained_logistic_with_ceiling_ramp,
    exponential_additions_with_ceiling,
)
from .learning_curve import capex_two_factor
from .finance import lcoe_series
from .ops_learning import ops_learning_factors, power_output_multiplier


def cf_path(years, cf0, cf1, ramp_years):
    """Capacity factor path with commissioning ramp."""
    y0 = years[0]
    cf = np.interp(years, [y0, y0 + ramp_years], [cf0, cf1], right=cf1)
    return np.clip(cf, 0.01, 0.95)


def run_once(cfg, rng_seed: int, with_fm: bool, share_mode="k"):
    """One-trial-one-draw Monte Carlo run.
    
    Each trial samples all uncertain parameters once, then runs the full pipeline:
    1. Experiments → compute time-to-gate → transform into schedule reduction
    2. Schedule draw → get FOAK months → compute ΔCOD
    3. Adoption: set t_mid from ΔCOD, apply supply-chain-constrained logistic
    4. Learning: compute CAPEX with two-factor curve + floor
    5. Ops & LCOE: build CF path, apply O&M multipliers, compute LCOE
    
    Args:
        cfg: configuration object
        rng_seed: random seed for reproducibility
        with_fm: whether to apply FM effects
        share_mode: "k" for k multiplier, "b" for b delta
    
    Returns:
        Dictionary with trial results
    """
    rng = np.random.default_rng(rng_seed)
    draw = sampled_config(cfg, rng)
    base = draw.base_config
    knobs = map_fm_to_knobs(base, draw, with_fm)
    assert_no_double_count(knobs, share_mode=share_mode)

    # Step 1: Experiments → schedule reduction
    exp = base.experiments
    if with_fm:
        s_pg, p_s = fm_adjust_experiment_inputs(exp, base.fm_effects.experiments)
    else:
        s_pg, p_s = exp.shots_per_gate, exp.shot_success_prob

    # Compute time-to-gate for base and FM scenarios
    t_gate_base = time_to_gate_days(
        exp.shots_per_gate,
        exp.shot_success_prob,
        exp.shots_per_day,
        exp.days_per_campaign,
        exp.days_between_campaigns,
    )
    t_gate_fm = time_to_gate_days(
        s_pg,
        p_s,
        exp.shots_per_day,
        exp.days_per_campaign,
        exp.days_between_campaigns,
    )
    
    # Convert experiment time savings to design time reduction
    design_reduction_from_exp = max(0.0, (t_gate_base - t_gate_fm) / max(1.0, t_gate_base))
    sched_knob = knobs["schedule"]
    sched_knob.design_time_reduction_pct = max(
        sched_knob.design_time_reduction_pct, design_reduction_from_exp
    )

    # Step 2: Schedule draw → get ΔCOD
    months_base = draw_schedule_months(base.schedule, None, rng)
    months_fm = draw_schedule_months(base.schedule, sched_knob, rng) if with_fm else months_base
    delta_years = (months_base - months_fm) / 12.0

    # Step 3: Adoption with schedule→adoption mapping
    A = base.adoption
    years = np.arange(max(2025, int(base.learning.t0)), 2071)
    
    if A.model == "bottom_up" and A.bottom_up_total_customers and A.bottom_up_build_years:
        N = bottom_up_N(
            years,
            A.bottom_up_total_customers,
            A.bottom_up_build_years,
            A.max_build_rate_per_year,
            N0=0.0,
        )
        t_mid = A.t_mid_base
        k = A.k_base
    elif A.model == "ceiling":
        # Pure exponential additions capped by a ceiling (no slowdown within horizon)
        start_adds = (A.ceiling_start_per_year or 0.5)
        growth = (A.ceiling_growth or 0.10) * (knobs["k_mult"] if with_fm else 1.0)
        N = exponential_additions_with_ceiling(
            years,
            start_adds,
            growth,
            ceiling_max_per_year=A.max_build_rate_per_year,
            N0=0.0,
        )
        # For reporting consistency, keep t_mid/k values
        t_mid = A.t_mid_base - (delta_years if with_fm else 0.0)
        k = A.k_base * (knobs["k_mult"] if with_fm else 1.0)
    else:
        # Schedule→adoption mapping: earlier schedule → earlier t_mid
        t_mid = A.t_mid_base - (delta_years if with_fm else 0.0)
        k = A.k_base * (knobs["k_mult"] if with_fm else 1.0)
        # Use ramping ceiling to avoid early spikes and mid-horizon slowdowns
        N = constrained_logistic_with_ceiling_ramp(
            years,
            A.n_max,
            k,
            t_mid,
            ceiling_start_per_year=0.5,   # ~0.5/year early 2025
            ceiling_growth=0.12,          # 12% annual growth in build capability
            ceiling_max_per_year=A.max_build_rate_per_year,
            N0=0.0,
        )

    # Step 4: Learning curve with two-factor model
    L = base.learning
    b = L.b_exponent + (knobs["b_delta"] if (with_fm and share_mode == "b") else 0.0)
    g = L.g_exogenous_per_year + (knobs["g_delta"] if with_fm else 0.0)
    capex = capex_two_factor(N, years, L.capex0_FOAK_USD, L.capex_floor_USD, b, g, L.N0, L.t0, getattr(L, "inertia_years", 0.0))
    assert_capex_monotone_when_g0(capex, g)

    # Step 5: Operations and LCOE
    O = base.ops
    cf_base_ramp = cf_path(years, O.cf_base_initial, O.cf_base_mature, O.commissioning_ramp_years)
    
    # Apply operational learning curve (CF improves, O&M reduces over time)
    # This accounts for:
    # - Capacity factor maturation (fewer disruptions, better uptime)
    # - Power output improvements (physics allows gradual gains over decades)
    # - O&M cost reductions (maintenance efficiency + economies of scale)
    
    # Set FM boost parameters
    fm_cf_boost = 0.4 if with_fm else 0.0  # FM accelerates CF/output gains by 40%
    fm_om_boost = 0.4 if with_fm else 0.0  # FM accelerates O&M reduction by 40%
    
    # Calculate operational learning factors (FM with more aggressive long-run gains)
    if with_fm:
        cf_learning, om_learning = ops_learning_factors(
            years,
            L.t0,
            tau_cf=50.0,
            tau_om=30.0,
            cf_fm_boost=fm_cf_boost,
            om_fm_boost=fm_om_boost,
            max_cf_gain=1.00,          # up to +100% power/output gain over long horizon
            max_om_reduction=0.80,     # up to -80% O&M reduction over long horizon
        )
    else:
        cf_learning, om_learning = ops_learning_factors(
            years,
            L.t0,
            tau_cf=50.0,
            tau_om=30.0,
            cf_fm_boost=0.0,
            om_fm_boost=0.0,
            max_cf_gain=0.50,          # up to +50% for baseline
            max_om_reduction=0.50,     # up to -50% for baseline
        )
    
    # Apply learning to base trajectory and include power output learning by scaling net MW
    cf = np.clip(cf_base_ramp * cf_learning, 0.01, 0.95)
    # O&M: apply FM control reduction only to the learned portion so both start equal
    # Base path (with learning only)
    fom = O.fom_base_per_year * om_learning
    vom = O.vom_base_per_MWh * om_learning
    
    # Apply FM control channel effects (additional step-change on top of learning)
    if with_fm:
        cf = np.clip(cf + knobs["cf_uplift"], 0.01, 0.95)
        # Instead of step change on full series, apply the reduction only to the learned portion
        # om_learning starts at 1.0 and decreases over time; (1 - om_learning) is the learned fraction
        fom = O.fom_base_per_year * om_learning * (1.0 - (1.0 - knobs["fom_mult"]) * (1.0 - om_learning))
        vom = O.vom_base_per_MWh * om_learning * (1.0 - (1.0 - knobs["vom_mult"]) * (1.0 - om_learning))
    
    assert_cf_bounds(cf)

    F = base.finance
    # Dynamic power output: separate learning multiplier (e.g., ~10% per decade baseline)
    # Keep CF path unchanged; let power improvements be slightly more aggressive
    power_mult = power_output_multiplier(
        years,
        L.t0,
        per_decade=0.10,
        fm_boost=(0.25 if with_fm else 0.0),  # FM boosts power learning to ~12.5%/decade
        cap=2.0,
    )
    net_power_series = base.meta.power_net_MW * power_mult
    lcoe = lcoe_series(capex, cf, fom, vom, F.wacc_real, F.life_years, net_power_series)

    return dict(
        seed=rng_seed,
        with_fm=with_fm,
        years=years,
        N=N,
        capex=capex,
        lcoe=lcoe,
        t_mid=t_mid,
        k=k,
        b=b,
        g=g,
        months_base=months_base,
        months_fm=months_fm,
        delta_cod_years=delta_years,
    )


