import numpy as np


def capex_two_factor(N, t, capex0, capex_floor, b, g, N0, t0, inertia_years: float = 0.0):
    """Two-factor learning curve with floor constraint.
    
    CAPEX(N,t) = max(CAPEX_floor, CAPEX_0 * (N/N_0)^(-b) * exp(-g*(t-t_0)))
    
    Args:
        N: cumulative reactor count by year
        t: years
        capex0: FOAK CAPEX ($)
        capex_floor: minimum CAPEX floor ($)
        b: learning exponent (learning rate = 1 - 2^(-b))
        g: exogenous annual progress rate
        N0: reference cumulative count
        t0: reference year
        inertia_years: optional smoothing horizon (default 0 = no smoothing)
    
    Returns:
        Array of CAPEX by year with floor constraint applied
    """
    N = np.asarray(N)
    t = np.asarray(t)
    
    # Apply inertia smoothing if specified
    if inertia_years and inertia_years > 0:
        # Simple EMA in time with half-life ~ inertia_years
        alpha = 1.0 - np.exp(-1.0 / max(1e-6, inertia_years))
        N_sm = np.empty_like(N, dtype=float)
        N_sm[0] = N[0]
        for i in range(1, len(N)):
            N_sm[i] = alpha * N[i] + (1 - alpha) * N_sm[i - 1]
        N_eff = N_sm
    else:
        N_eff = N
    
    # Ensure N_eff >= N0 to avoid negative exponents
    N_eff = np.maximum(N_eff, N0)
    
    # Two-factor learning curve
    capex = capex0 * (N_eff / N0) ** (-b) * np.exp(-g * (t - t0))
    
    # Apply floor constraint
    capex = np.maximum(capex, capex_floor)
    
    return capex


