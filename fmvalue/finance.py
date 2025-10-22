import numpy as np


def crf(r: float, n: int) -> float:
    """Capital Recovery Factor: CRF = r(1+r)^n / ((1+r)^n - 1)
    
    Args:
        r: real WACC (discount rate)
        n: project life in years
    
    Returns:
        CRF for converting CAPEX to annualized cost
    """
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def lcoe_series(capex, cf, fom_py, vom_pmwh, wacc, life_years, net_MW_or_series):
    """Finance-correct LCOE calculation with dynamic power output.

    net_MW_or_series may be a scalar or an array (e.g., reflecting power output learning).
    """
    capex = np.asarray(capex)
    cf = np.asarray(cf)
    net_series = np.asarray(net_MW_or_series) if np.ndim(net_MW_or_series) else net_MW_or_series

    crf_val = crf(wacc, life_years)
    annualized_capex = crf_val * capex

    # Energy generation per year (MWh)
    energy_MWh = 8760 * cf * net_series

    capex_component = annualized_capex / energy_MWh
    fom_component = fom_py / energy_MWh
    vom_component = vom_pmwh

    lcoe = capex_component + fom_component + vom_component
    return lcoe


