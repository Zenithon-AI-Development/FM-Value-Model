"""Demand-driven reactor deployment model.

Bottom-up logic:
1. Global electricity demand grows ~2%/yr (baseline projection)
2. Fusion becomes competitive when LCOE < coal/gas (~$50-80/MWh)
3. Market share captured via S-curve once competitive
4. Annual reactor additions = (target capacity - current) / build_time + replacement
"""
import numpy as np


def electricity_demand_TWh(years, base_year=2030, base_demand_TWh=28000, growth_rate=0.02):
    """Global electricity demand projection (IEA-style growth).
    
    Args:
        years: array of years
        base_year: reference year
        base_demand_TWh: demand in base year (TWh/yr)
        growth_rate: annual growth rate (e.g., 0.02 = 2%/yr)
    
    Returns:
        Array of demand in TWh/yr
    """
    years = np.asarray(years)
    return base_demand_TWh * np.exp(growth_rate * (years - base_year))


def fusion_market_share(years, lcoe_fusion, lcoe_competitor=60.0, 
                         t_competitive=2035, k=0.15, max_share=0.15):
    """Fusion market penetration S-curve once competitive.
    
    Args:
        years: array of years
        lcoe_fusion: array of fusion LCOE ($/MWh) by year
        lcoe_competitor: competing generation cost ($/MWh)
        t_competitive: year when fusion first becomes competitive
        k: diffusion rate (logistic steepness)
        max_share: maximum market share fusion can capture (supply/policy limits)
    
    Returns:
        Array of market share (0-1) by year
    """
    years = np.asarray(years)
    lcoe_fusion = np.asarray(lcoe_fusion)
    
    # Only penetrate market where fusion is cheaper than competitor
    competitive = lcoe_fusion < lcoe_competitor
    
    # S-curve starting from first competitive year
    t_start = years[competitive][0] if np.any(competitive) else years[-1]
    t_mid = t_start + 10  # reach 50% of max_share 10 years after becoming competitive
    
    # Logistic curve
    share = max_share / (1.0 + np.exp(-k * (years - t_mid)))
    
    # Zero out years before competitive
    share[years < t_start] = 0.0
    
    return share


def reactor_buildout_from_demand(years, demand_TWh, market_share, 
                                   reactor_capacity_GW=1.0, capacity_factor=0.80):
    """Convert market demand to number of reactors needed.
    
    Args:
        years: array of years
        demand_TWh: global electricity demand (TWh/yr)
        market_share: fusion market share (0-1)
        reactor_capacity_GW: nameplate capacity per reactor (GW)
        capacity_factor: average CF
    
    Returns:
        Cumulative number of reactors needed
    """
    # Fusion generation target (TWh/yr)
    fusion_generation_TWh = demand_TWh * market_share
    
    # Convert to GW average (8760 hr/yr)
    fusion_capacity_avg_GW = fusion_generation_TWh * 1000 / 8760
    
    # Nameplate capacity needed (accounting for CF)
    fusion_capacity_nameplate_GW = fusion_capacity_avg_GW / capacity_factor
    
    # Number of reactors
    N_reactors = fusion_capacity_nameplate_GW / reactor_capacity_GW
    
    return N_reactors


def annual_additions(N_cumulative, replace_after_years=40):
    """Compute annual reactor additions from cumulative trajectory.
    
    Includes replacement demand for reactors reaching end-of-life.
    
    Args:
        N_cumulative: array of cumulative reactors by year
        replace_after_years: reactor lifetime (years)
    
    Returns:
        Array of annual additions (new + replacement)
    """
    N_cumulative = np.asarray(N_cumulative)
    
    # New additions (diff of cumulative)
    new = np.diff(np.r_[0, N_cumulative])
    new = np.maximum(new, 0)  # no negative builds
    
    # Replacement demand (reactors from `replace_after_years` ago retiring)
    replacement = np.zeros_like(new)
    for i in range(len(new)):
        if i >= replace_after_years:
            replacement[i] = new[i - replace_after_years]
    
    return new + replacement

