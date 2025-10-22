# Foundation Model for Fusion Value Estimates

Quantifies the value of a fusion Foundation Model (FM) by tracing a transparent, non‑overlapping causal chain from R&D to operations to economics. Produces CAPEX/LCOE trajectories, parity timing, total CAPEX and savings, and total deployed fusion power.

## Quick start

1) Create an environment and install dependencies

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run the notebook end‑to‑end

```bash
cd fm_value
jupyter lab  # or jupyter notebook
# Open fm_value.ipynb and run all cells top-to-bottom
```

3) Tweak assumptions in one place

Edit `fm_value/inputs.yaml`. It is the single source of truth for:
- Finance (WACC, life)
- Operations (FOAK/mature CF, FOM/VOM, commissioning ramp)
- Schedule (design/EPC/commissioning modes, rework, uplift)
- Experiments (shots per gate, success rate, cadence)
- Adoption (supply‑chain ceiling: start rate, growth, max build rate)
- Learning (FOAK CAPEX, floor, exponent b, exogenous rate g, inertia)
- FM effects (simulation, experiments, control, sharing — one primary lever each)

The notebook reads this file to produce deterministic paths and Monte Carlo uncertainty bands.

## How it works (end‑to‑end)

1) Experiments → schedule
- Compute time‑to‑gate from shots and success probability
- FM reduces shots and increases success; mapped to a design‑time reduction
- Schedule is drawn with rework risks and reference‑class uplift

2) Schedule → adoption timing
- Shorter FOAK → earlier COD → earlier adoption inflection
- Deployment uses a supply‑driven “ceiling ramp”: annual additions grow exponentially from `ceiling_start_per_year` at `ceiling_growth`, capped by `max_build_rate_per_year`
- FM sharing steepens adoption via a single multiplier (no double count)

3) Adoption → learning → CAPEX
- Two‑factor learning with floor:
  - CAPEX(N,t) = max(floor, FOAK × (N/N0)^(-b) × exp(-g·Δt))
  - Optional inertia smoothing slows response to rapid N changes

4) Operations learning and control → CF, power, O&M
- CF path = commissioning ramp × long‑horizon saturation multiplier; FM adds bounded control uplift
- Power‑output learning is separate from CF (e.g., ~10%/decade baseline, modest FM boost)
- O&M reductions follow saturation; FM reductions apply only to the learned portion so base and FM start equal and diverge over time

5) Finance → LCOE
- LCOE = (CRF×CAPEX + FOM) / (8760 × CF × net_power) + VOM
- CRF from WACC & life; `net_power` is dynamic (power‑output learning)

Outputs: time series for N(t), CAPEX(t), CF(t), power(t), O&M(t), LCOE(t); summary metrics (total CAPEX and savings, LCOE@2050, parity year, total fusion power@2050).

## Monte Carlo uncertainty

The notebook runs a one‑trial‑one‑draw MC: each trial samples priors (e.g., WACC, learning b, floors, schedule risks) and reproduces the full pipeline. Ribbons show 5–95% quantiles. Deterministic lines use central inputs; bands are centered to visually track those lines.

## Key knobs to try

- Adoption ceiling
  - `adoption.ceiling_start_per_year` (initial build capability)
  - `adoption.ceiling_growth` (annual growth of capability)
  - `adoption.max_build_rate_per_year` (cap on annual additions)
- Learning
  - `learning.b_exponent` (experience curve steepness)
  - `learning.g_exogenous_per_year` (exogenous progress)
  - `learning.inertia_years` (smoothing)
- Operations
  - `ops.cf_base_initial`, `ops.cf_base_mature`, `ops.commissioning_ramp_years`
  - `ops.fom_base_per_year`, `ops.vom_base_per_MWh`
- FM effects (single primary lever per channel; no double counting)

## Repository layout

```
fm_value/
  fm_value.ipynb       # main notebook: deterministic + MC ribbons + metrics
  inputs.yaml          # all assumptions and priors
fmvalue/
  mc.py                # one-trial-one-draw Monte Carlo
  learning_curve.py    # CAPEX learning (two-factor + floor + inertia)
  ops_learning.py      # ops & power learning curves
  adoption.py          # adoption ceiling models
  schedule.py          # schedule draw with uplift
  experiments.py       # shots-to-gate mapping
  finance.py           # CRF + LCOE
  models.py            # Pydantic config models
  viz.py, checks.py    # plotting helpers, acceptance checks
```

## Troubleshooting

- Validation: `commissioning_ramp_years` must be ≤ 10 (see `fmvalue/models.py`). If you want longer ramps, lower the YAML value or relax the constraint in the model.
- Bands not tracking lines: ensure the notebook is run top‑to‑bottom after any YAML change so deterministic paths and MC ribbons use the same inputs.
