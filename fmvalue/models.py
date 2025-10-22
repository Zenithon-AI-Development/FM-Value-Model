from pydantic import BaseModel, Field
from typing import Literal, Optional


class FinanceCfg(BaseModel):
    wacc_real: float = Field(..., ge=0.0, le=0.2)
    life_years: int = Field(..., ge=10, le=60)


class OpsCfg(BaseModel):
    cf_base_initial: float = Field(..., ge=0.01, le=0.95)
    cf_base_mature: float = Field(..., ge=0.01, le=0.95)
    fom_base_per_year: float = Field(..., ge=0.0)
    vom_base_per_MWh: float = Field(..., ge=0.0)
    commissioning_ramp_years: int = Field(..., ge=0, le=12)


class ScheduleCfg(BaseModel):
    # Modes provided but priors will override to sampled fields (design_months, etc.)
    design_months_mode: float
    epc_months_mode: float
    commission_months_mode: float
    # Sampled fields injected by sampling step
    design_months: Optional[float] = None
    epc_months: Optional[float] = None
    commission_months: Optional[float] = None
    rework_prob: float = Field(..., ge=0.0, le=1.0)
    rework_factor: Optional[float] = None
    epc_rework_tail_prob: float = Field(..., ge=0.0, le=1.0)
    epc_rework_tail_factor: float = Field(..., ge=0.0, le=1.0)
    uplift_lognorm_mu: float
    uplift_lognorm_sigma: float


class ExperimentsCfg(BaseModel):
    shots_per_gate: int
    shot_success_prob: float = Field(..., ge=0.0, le=1.0)
    shots_per_day: int
    days_per_campaign: int
    days_between_campaigns: int




class FMAdoptionCfg(BaseModel):
    """FM adoption among fusion companies (separate from reactor count)."""
    total_companies: int = Field(..., ge=1)  # e.g., 50 fusion companies
    t_mid: float  # year when 50% of companies adopt FM
    k: float = Field(..., ge=0.01)  # diffusion rate
    
    
class AdoptionCfg(BaseModel):
    """Adoption model configuration.

    Supported models:
    - logistic: logistic with supply ceiling
    - bottom_up: deterministic bottom-up
    - ceiling: ceiling-driven exponential additions (no slowdown within horizon)
    """
    model: Literal["logistic", "bass", "bottom_up", "ceiling"] = "logistic"
    t_mid_base: float = 2045.0
    k_base: float = Field(default=0.20, ge=0.01)
    n_max: int = Field(default=300, ge=1)
    max_build_rate_per_year: int = Field(default=20, ge=1)
    # Optional ceiling ramp parameters (used when model == "ceiling")
    ceiling_start_per_year: Optional[float] = None
    ceiling_growth: Optional[float] = None
    # Bottom-up optional parameters (used when model == "bottom_up")
    bottom_up_total_customers: Optional[int] = None
    bottom_up_build_years: Optional[int] = None


class LearningCfg(BaseModel):
    capex0_FOAK_USD: float = Field(..., ge=1e9)
    capex_floor_USD: float = Field(..., ge=1e9)
    b_exponent: float = Field(..., ge=0.0, le=1.0)
    g_exogenous_per_year: float
    N0: float = 1.0
    t0: float
    inertia_years: Optional[float] = 0.0  # smoothing horizon for learning response


class FMEffectsSim(BaseModel):
    delta_g_per_year: float = 0.0
    design_time_reduction_pct: float = 0.0
    rework_prob_reduction_pct: float = 0.0


class FMEffectsExp(BaseModel):
    shots_reduction_pct: float = 0.0
    success_prob_uplift: float = 0.0


class FMEffectsCtrl(BaseModel):
    cf_uplift_abs: float = 0.0
    fom_reduction_pct: float = 0.0
    vom_reduction_pct: float = 0.0


class FMEffectsShare(BaseModel):
    delta_b_exponent: float = 0.0
    k_multiplier: float = 1.0


class FMEffectsCfg(BaseModel):
    simulation: FMEffectsSim
    experiments: FMEffectsExp
    control: FMEffectsCtrl
    sharing: FMEffectsShare


class MetaCfg(BaseModel):
    currency: str
    power_net_MW: float


class BaseConfig(BaseModel):
    meta: MetaCfg
    finance: FinanceCfg
    ops: OpsCfg
    schedule: ScheduleCfg
    experiments: ExperimentsCfg
    fm_adoption: FMAdoptionCfg
    adoption: AdoptionCfg
    learning: LearningCfg
    fm_effects: FMEffectsCfg


class PriorDef(BaseModel):
    dist: Literal["triangular", "lognormal", "constant"]
    params: list[float]


class Config(BaseModel):
    base_config: BaseConfig
    priors: dict[str, PriorDef]
    sources: dict[str, list[str]]


