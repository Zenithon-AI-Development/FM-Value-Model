from dataclasses import dataclass


@dataclass
class FMSchedule:
    design_time_reduction_pct: float = 0.0
    rework_prob_reduction_pct: float = 0.0


def map_fm_to_knobs(cfg_base, cfg_drawn, with_fm: bool):
    sim = cfg_base.fm_effects.simulation if with_fm else None
    ctrl = cfg_base.fm_effects.control if with_fm else None
    share = cfg_base.fm_effects.sharing if with_fm else None

    knobs = {
        "schedule": FMSchedule(
            design_time_reduction_pct=(sim.design_time_reduction_pct if sim else 0.0),
            rework_prob_reduction_pct=(sim.rework_prob_reduction_pct if sim else 0.0),
        ),
        "g_delta": (sim.delta_g_per_year if sim else 0.0),
        "b_delta": (share.delta_b_exponent if share else 0.0),
        "k_mult": (share.k_multiplier if share else 1.0),
        "cf_uplift": (ctrl.cf_uplift_abs if ctrl else 0.0),
        "fom_mult": (1.0 - (ctrl.fom_reduction_pct if ctrl else 0.0)),
        "vom_mult": (1.0 - (ctrl.vom_reduction_pct if ctrl else 0.0)),
    }
    return knobs


