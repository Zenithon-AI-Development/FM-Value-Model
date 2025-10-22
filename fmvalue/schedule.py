import numpy as np


def draw_schedule_months(sched_cfg, fm_sched, rng: np.random.Generator):
    design = sched_cfg.design_months if getattr(sched_cfg, "design_months", None) is not None else sched_cfg.design_months_mode
    if fm_sched:
        design *= (1.0 - fm_sched.design_time_reduction_pct)

    rework_prob_eff = sched_cfg.rework_prob * (1.0 - (fm_sched.rework_prob_reduction_pct if fm_sched else 0.0))
    if rng.random() < rework_prob_eff:
        rework_factor = sched_cfg.rework_factor if getattr(sched_cfg, "rework_factor", None) is not None else 0.1
        design *= (1.0 + rework_factor)

    epc = sched_cfg.epc_months if getattr(sched_cfg, "epc_months", None) is not None else sched_cfg.epc_months_mode
    if rng.random() < sched_cfg.epc_rework_tail_prob:
        epc *= (1.0 + sched_cfg.epc_rework_tail_factor)

    comm = sched_cfg.commission_months if getattr(sched_cfg, "commission_months", None) is not None else sched_cfg.commission_months_mode
    months = design + epc + comm

    uplift = np.random.lognormal(
        mean=np.log(1 + sched_cfg.uplift_lognorm_mu) - 0.5 * sched_cfg.uplift_lognorm_sigma ** 2,
        sigma=sched_cfg.uplift_lognorm_sigma,
    )
    return months * uplift


