def assert_no_double_count(fm_knobs, share_mode="k"):
    if share_mode == "k":
        assert abs(fm_knobs["b_delta"]) < 1e-12, "Sharing must not change b when using k_multiplier."
    if share_mode == "b":
        assert abs(fm_knobs["k_mult"] - 1.0) < 1e-12, "Sharing must not change k when using b_delta."


def assert_capex_monotone_when_g0(capex, g):
    if abs(g) < 1e-9:
        diffs = (capex[1:] - capex[:-1])
        assert (diffs <= 1e-6).all(), "CAPEX increased with g=0; check N(t) or floor interplay."


def assert_cf_bounds(cf):
    import numpy as np
    assert (cf >= 0.01).all() and (cf <= 0.95).all(), "CF out of bounds."


