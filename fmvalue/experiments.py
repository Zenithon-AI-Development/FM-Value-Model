import math


def time_to_gate_days(
    shots_per_gate: float,
    p_succ: float,
    shots_per_day: float,
    days_per_campaign: int,
    days_between_campaigns: int,
):
    eff_shots = shots_per_gate / max(1e-6, p_succ)
    days_shots = eff_shots / shots_per_day
    campaigns = math.ceil(days_shots / days_per_campaign)
    return days_shots + max(0, campaigns - 1) * days_between_campaigns


def fm_adjust_experiment_inputs(exp_cfg, fm_exp):
    shots = exp_cfg.shots_per_gate * (1.0 - fm_exp.shots_reduction_pct)
    p_succ = min(0.99, exp_cfg.shot_success_prob + fm_exp.success_prob_uplift)
    return shots, p_succ


