import numpy as np
import pandas as pd


def year_when(df: pd.DataFrame, target: float, key="lcoe"):
    """Find first year when median metric reaches target value."""
    med = df.groupby("year")[key].median()
    hit = med[med <= target]
    return None if hit.empty else int(hit.index[0])


def acceptance_checks(df_base: pd.DataFrame, df_fm: pd.DataFrame):
    """Acceptance checks per specification.
    
    Sanity anchors (not hard targets):
    - CAPEX~$10B in 2030; trending to ~$5B by 2050
    - Baseline LCOE approaches $50/MWh by mid-century
    - With-FM should reach 5¢ earlier (few years) vs baseline median
    """
    med = lambda df, y, k: float(df[(df.year == y)][k].median())
    
    # CAPEX anchors
    capex_2030 = med(df_base, 2030, "capex")
    capex_2050 = med(df_base, 2050, "capex")
    lcoe_2050 = med(df_base, 2050, "lcoe")

    # CAPEX checks with reasonable tolerance
    assert 7e9 <= capex_2030 <= 13e9, f"CAPEX@2030 too far from ~$10B anchor: ${capex_2030/1e9:.1f}B"
    assert 3.5e9 <= capex_2050 <= 6.0e9, f"CAPEX@2050 not trending near ~$5B: ${capex_2050/1e9:.1f}B"
    assert lcoe_2050 <= 60.0, f"Baseline LCOE not approaching ~$50/MWh: ${lcoe_2050:.1f}/MWh"

    # FM impact check: should reach 5¢/kWh earlier
    base_t50 = year_when(df_base, 50.0, key="lcoe")
    fm_t50 = year_when(df_fm, 50.0, key="lcoe")
    if base_t50 and fm_t50:
        assert fm_t50 <= base_t50 - 2, f"With-FM should reach 5¢ earlier by a few years: base={base_t50}, fm={fm_t50}"


