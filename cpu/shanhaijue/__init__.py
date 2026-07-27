# learning AI at www.haotianblog.com
"""ShanHaiJue + ShanHaiJian: routed multi-estimator identification and risk-controlled
certification for battery EIS parameter recovery (CPU distribution).

Quick start
-----------
    from shanhaijue import panel, arms, features, router, certifier

    truth = panel.draw_truth(0, seed_base=20260101)
    z     = panel.simulate_observation(truth, seed=1)
    a, aux = arms.fit_ensemble(z, seed=1)                  # 8 estimator arms
    rows  = features.extract(a, aux, z, truth=truth, cell_id=(0, 0))
    # ... train a router on many cells, then freeze and certify (see bin/).

Pipeline stages
---------------
    panel      forward model, parameter space, synthetic panel generation
    arms       the estimator ensemble along the prior-strength axis
    features   ten ground-truth-free per-coordinate diagnostics
    router     gradient-boosted ranker + per-arm calibration + arg-max routing
    certifier  Platt calibration + cluster-UCB false-claim budget control
"""
__version__ = "1.0.0"
__all__ = ["panel", "arms", "features", "router", "certifier", "pipeline"]
