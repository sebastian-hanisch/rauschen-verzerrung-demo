"""Die vorgerechnete Messreihe (data/messreihe.json, Erzeuger tools/make_messreihe.py): vollständig, zur Konstanten-Datei passend, ohne tote Zellen."""

import math

import numpy as np
import pytest

import rau_constants as C
import rau_evaluation as EV

DATA = EV.load_messreihe()


def test_meta_matches_the_constants():
    meta = DATA["meta"]
    assert meta["runs"] == C.SWEEP_RUNS and meta["seeds"] == list(C.SWEEP_SEEDS)
    assert (meta["plan_n"], meta["plan_gap"], meta["perf_n"], meta["judges"], meta["assignments"]) == (C.DEFAULT_PLAN_N, C.DEFAULT_PLAN_GAP, C.DEFAULT_PERF_N, C.DEFAULT_JUDGES, C.PLAN_ASSIGNMENTS)


def test_every_cell_exists_with_five_seeds():
    for m in C.SWEEP_FORM_M:
        for rho in C.SWEEP_FORM_RHO:
            assert DATA["plan"][f"m={m}|rho={rho}"]["seed"] == list(C.SWEEP_SEEDS)
    for b in C.SWEEP_DRIFT:
        assert DATA["auftritt_drift"][f"{b}"]["seed"] == list(C.SWEEP_SEEDS)
    for g in C.SWEEP_ANCHOR:
        assert DATA["auftritt_anchor"][f"{g}"]["seed"] == list(C.SWEEP_SEEDS)
    assert len(DATA["plan"]) == len(C.SWEEP_FORM_M) * len(C.SWEEP_FORM_RHO)


def test_no_dead_column_every_sweep_with_a_distortion_has_a_nonzero_lage_difference():
    """Nullspalten sind ein Fehlersignal (DEMO-PLAYBOOK §0): ohne Verzerrung ist Δ null, mit Verzerrung in jeder Zelle und jedem Zufalls-Seed positiv."""
    for rho in C.SWEEP_FORM_RHO:
        for m in C.SWEEP_FORM_M:
            deltas = DATA["plan"][f"m={m}|rho={rho}"]["delta"]
            if m == 0:
                assert deltas == [0.0] * 5
            else:
                assert all(d > 0 for d in deltas), (m, rho)
    assert DATA["auftritt_drift"]["0.0"]["delta"] == [0.0] * 5
    for key in C.SWEEP_DRIFT[1:]:
        assert all(d > 0 for d in DATA["auftritt_drift"][f"{key}"]["delta"])
    for g in C.SWEEP_ANCHOR:
        assert all(d > 0 for d in DATA["auftritt_anchor"][f"{g}"]["delta"])


def test_the_chances_are_probabilities_and_consistent():
    for family in ("plan", "auftritt_drift", "auftritt_anchor"):
        for cell in DATA[family].values():
            for pg, pb, d in zip(cell["p_good"], cell["p_bad"], cell["delta"]):
                assert 0 < pb <= 1 and 0 < pg <= 1 and abs((pg - pb) - d) < 2e-6


def test_the_variation_between_seeds_is_reported_and_finite():
    for rho in C.SWEEP_FORM_RHO:
        for row in EV.sweep_plan(rho):
            assert math.isfinite(row["spread"]) and row["spread"] >= 0
    assert all(math.isfinite(r["spread"]) for r in EV.sweep_judges_drift())


def test_every_sweep_row_set_has_the_documented_length():
    assert len(EV.sweep_plan(0.5)) == len(C.SWEEP_FORM_M) and len(EV.sweep_judges_drift()) == len(C.SWEEP_DRIFT) and len(EV.sweep_judges_anchor()) == len(C.SWEEP_ANCHOR)
    assert len(EV.sweep_duel(False)) == len(C.SWEEP_LAM_HOME) and len(EV.sweep_shoot("ABAB")) == len(C.SWEEP_PEN_D)


def test_the_mean_over_seeds_is_the_value_the_app_uses():
    cell = DATA["plan"]["m=100|rho=0.5"]
    row = {r["x"]: r for r in EV.sweep_plan(0.5)}[100]
    assert row["delta"] == pytest.approx(float(np.mean(cell["delta"])))
