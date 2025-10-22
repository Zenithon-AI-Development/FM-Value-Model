import copy
import numpy as np
from .models import Config


def sample_from_prior(pr, rng: np.random.Generator):
    # Accept either dict-like or Pydantic PriorDef
    if hasattr(pr, "dist") and hasattr(pr, "params"):
        d = pr.dist
        p = pr.params
    else:
        d = pr["dist"]
        p = pr["params"]
    if d == "triangular":
        a, m, b = p
        return rng.triangular(a, m, b)
    if d == "lognormal":
        mu, sigma = p
        return rng.lognormal(mean=mu, sigma=sigma)
    if d == "constant":
        return p[0]
    raise ValueError(f"Unknown dist {d}")


def _set_by_path(dct: dict, path: str, value):
    parts = path.split(".")
    ref = dct
    for p in parts[:-1]:
        ref = ref[p]
    ref[parts[-1]] = value


def sampled_config(cfg: Config, rng: np.random.Generator) -> Config:
    data = cfg.model_dump()
    for key, prior in cfg.priors.items():
        value = sample_from_prior(prior, rng)
        # Coerce integer-typed fields expected by the model
        if key in {
            "experiments.shots_per_gate",
            "experiments.shots_per_day",
            "experiments.days_per_campaign",
            "experiments.days_between_campaigns",
        }:
            value = int(round(value))
        _set_by_path(data["base_config"], key, value)
    return Config(**data)


