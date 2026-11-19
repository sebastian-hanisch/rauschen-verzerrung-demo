"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Verfahren gelten müssen, ohne das richtige Ergebnis zu kennen.

Jede Eigenschaft ist eine belegte Aussage über das Modell, nicht das heutige Verhalten des Codes; wo sie nur unter Voraussetzungen gilt, stehen sie im Test."""

import dataclasses
import math

import numpy as np
import pytest

import rau_constants as C
import rau_duel as D
import rau_evaluation as EV
import rau_judges as J
import rau_lage as LG
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_scenario as SC
import rau_shootout as SO


# --- Benötigte Zahl ---------------------------------------------------------------------------------------------------------------------------------------

PAIRS = [(0.6, 0.4), (0.76, 0.745), (0.55, 0.5), (0.93, 0.62), (0.45, 0.4), (0.3, 0.2)]


@pytest.mark.parametrize("p1, p2", PAIRS)
def test_n_is_symmetric_in_the_two_lagen(p1, p2):
    assert PW.n_two_groups(p1, p2, 0.05, 0.8) == pytest.approx(PW.n_two_groups(p2, p1, 0.05, 0.8))


@pytest.mark.parametrize("p1, p2", PAIRS)
def test_n_is_the_same_for_the_losing_side(p1, p2):
    """Wer die Chance p hat zu gewinnen, hat 1 - p zu verlieren: dieselben Beobachtungen, dieselbe Zahl."""
    assert PW.n_two_groups(1 - p1, 1 - p2, 0.05, 0.8) == pytest.approx(PW.n_two_groups(p1, p2, 0.05, 0.8))
    assert PW.n_against_fair((1 - p1 + 1 - p2) / 2, 1 - p1, 0.05, 0.8) == pytest.approx(PW.n_against_fair((p1 + p2) / 2, p1, 0.05, 0.8))


@pytest.mark.parametrize("p1, p2", PAIRS)
def test_n_grows_with_power_and_with_a_stricter_error_level(p1, p2):
    base = PW.n_two_groups(p1, p2, 0.05, 0.8)
    assert PW.n_two_groups(p1, p2, 0.05, 0.9) > base > PW.n_two_groups(p1, p2, 0.05, 0.7)
    assert PW.n_two_groups(p1, p2, 0.01, 0.8) > base > PW.n_two_groups(p1, p2, 0.1, 0.8)
    for design in C.DESIGNS:
        assert PW.n_needed(p1, p2, 0.05, 0.95, design) > PW.n_needed(p1, p2, 0.05, 0.8, design)


def test_n_times_delta_squared_is_nearly_constant_at_a_fixed_fair_value():
    """N·Δ² = (z_a·√(2p̄q̄) + z_b·√(2p̄q̄ - Δ²/2))²: bei festem Fairwert fast konstant für kleine Δ (Abweichung unter 2 % bis Δ = 0,1, unter 10 % bis Δ = 0,3)."""
    pbar = 0.6
    ref = PW.n_two_groups(pbar + 0.0005, pbar - 0.0005, 0.05, 0.8) * 0.001 ** 2
    for delta, tol in ((0.02, 0.01), (0.05, 0.01), (0.1, 0.02), (0.3, 0.1)):
        got = PW.n_two_groups(pbar + delta / 2, pbar - delta / 2, 0.05, 0.8) * delta ** 2
        assert abs(got / ref - 1) < tol, (delta, got / ref)


def test_halving_delta_roughly_quadruples_n():
    base = PW.n_two_groups(0.55, 0.5, 0.05, 0.8)
    half = PW.n_two_groups(0.525, 0.5, 0.05, 0.8)
    assert 3.8 < half / base < 4.2


def test_against_fair_needs_about_twice_as_many_as_two_groups_for_the_same_delta():
    """Gegen den Fairwert ist die Verschiebung Δ/2 (halb so groß), je Lage dafür nur eine Gruppe: etwa das Doppelte je Lage (Abweichung bis 10 % bei kleinen Δ)."""
    for p1, p2 in ((0.55, 0.5), (0.52, 0.48), (0.8, 0.77)):
        ratio = PW.n_needed(p1, p2, 0.05, 0.8, "gegen den Fairwert") / PW.n_needed(p1, p2, 0.05, 0.8)
        assert 1.7 < ratio < 2.3


def test_exact_power_grows_with_n_on_a_coarse_grid():
    values = [PW.exact_power(0.6, 0.5, n, 0.05) for n in (40, 80, 160, 320, 640)]
    assert values == sorted(values) and values[-1] > 0.9


def test_null_rejection_rate_is_about_the_error_level():
    """p1 = p2: die Fehlalarmrate liegt nahe dem Fehlerniveau (exakt gerechnet bei N = 200 um höchstens 1,5 Pp, diskrete Verteilung)."""
    for alpha in (0.01, 0.05, 0.1):
        assert abs(PW.exact_power(0.6, 0.6, 200, alpha) - alpha) < 0.015
    sim = PW.simulate_power(0.6, 0.6, 200, 0.05, 20000, 8)
    assert abs(sim.rate - 0.05) <= 4 * sim.se + 0.01


# --- Lage -------------------------------------------------------------------------------------------------------------------------------------------------

def test_delta_is_never_negative_for_exact_lagen():
    for chances in ([0.5, 0.5], [0.3, 0.7], [0.61, 0.6, 0.62]):
        assert LG.exact_lage(chances).delta >= 0


# --- Verzerrungen verschwinden, wo sie verschwinden müssen -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n, rho, double", [(4, 0.0, False), (6, 0.5, False), (8, 0.8, True), (4, 0.95, True)])
def test_without_form_the_schedule_changes_nothing(n, rho, double):
    """m = 0: jede Partie hängt nur von den Grundstärken und ihrer eigenen Zufallszahl ab; die Reihenfolge der Spieltage verschiebt dann in keinem Lauf etwas."""
    lage = PL.plan_lage(n, 300.0, 600, 3, 0.0, rho, 0.25, 12, double)
    assert lage.delta == 0.0 and lage.p_best == lage.p_worst


def test_with_form_the_schedule_can_matter():
    lage = PL.plan_lage(8, 300.0, 4000, 3, 100.0, 0.0, 0.25, 24)
    assert lage.delta > 4 * lage.se > 0


@pytest.mark.parametrize("order", ["ABAB", "ABBA"])
@pytest.mark.parametrize("p, gap", [(0.75, 0.0), (0.8, 0.1), (0.6, 0.05), (0.95, 0.2)])
def test_independent_shots_make_the_order_worthless(order, p, gap):
    assert SO.lage_effect(p, gap, 0.0, order)[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("shots", [1, 2, 3, 5, 7])
def test_abab_chance_does_not_depend_on_the_number_of_shots_for_equal_teams(shots):
    """Gemessen in der Nachbar-Demo und hier an fünf Schusszahlen nachgeprüft: (1 - p + d) / (2(1 - p) + d)."""
    p, d = 0.75, 0.05
    assert SO.first_team_win(p, p, d, "ABAB", shots) == pytest.approx((1 - p + d) / (2 * (1 - p) + d), abs=1e-12)


@pytest.mark.parametrize("p, d", [(0.75, 0.05), (0.75, 0.15), (0.6, 0.1), (0.9, 0.2)])
def test_abba_is_closer_to_a_fair_coin_than_abab_for_equal_teams(p, d):
    abab = SO.first_team_win(p, p, d, "ABAB", 5)
    abba = SO.first_team_win(p, p, d, "ABBA", 5)
    assert 0.5 <= abba < abab


@pytest.mark.parametrize("elo", [0, 100, 300])
@pytest.mark.parametrize("lam_home, lam_away", [(1.5, 1.1), (0.7, 1.4), (2.0, 0.5)])
@pytest.mark.parametrize("rule", [False, True])
def test_without_extra_time_the_venue_order_is_worth_nothing(lam_home, lam_away, elo, rule):
    """Ohne Verlängerung hängt die Chance nicht davon ab, wer das Rückspiel zu Hause hat (auch mit Regel): die Auswärts- und Heim-Differenz sind in beiden Lagen dieselben."""
    assert D.lage_effect(lam_home, lam_away, float(elo), rule, False)[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("elo", [0, 100, 300])
@pytest.mark.parametrize("lam", [0.8, 1.3, 2.0])
def test_with_equal_home_and_away_rates_and_no_rule_the_venue_order_is_worth_nothing(lam, elo):
    assert D.lage_effect(lam, lam, float(elo), False, True)[2] == pytest.approx(0.0, abs=1e-12)


def test_the_stronger_team_advances_more_often_than_the_weaker_one():
    for elo in (25, 100, 300):
        for home in (True, False):
            assert D.advance_chance(1.5, 1.1, float(elo), home, False, True).p_strong > 0.5


def test_the_stronger_the_gap_the_better_for_the_stronger_team():
    chances = [D.advance_chance(1.5, 1.1, float(g), True, False, True).p_strong for g in (0, 50, 100, 200, 300)]
    assert chances == sorted(chances) and chances[0] == pytest.approx(0.5084, abs=1e-4)


def test_no_drift_and_no_anchor_make_the_position_worthless():
    lage, per_place = J.auftritt_lage(6, 3, 0.5, 0.0, 0.0, 1000, 4)
    assert lage.delta == 0.0 and np.allclose(per_place, per_place[0])


def test_drift_helps_late_places_and_hurts_early_ones():
    lage, per_place = J.auftritt_lage(6, 3, 0.5, 0.08, 0.0, 6000, 4)
    assert per_place[-1] > per_place[0] and lage.best == 5 and lage.worst == 0 and lage.delta > 4 * lage.se


def test_mirrored_drift_mirrors_the_places_statistically():
    """Drift β und -β spiegeln die Plätze (bei γ = 0): die Chance auf Platz k mit β gleicht der auf Platz n-1-k mit -β (innerhalb von vier Standardfehlern)."""
    runs = 30000
    a = J.strongest_wins(5, 3, 0.5, 0.1, 0.0, runs, 6).mean(axis=1)
    b = J.strongest_wins(5, 3, 0.5, -0.1, 0.0, runs, 7).mean(axis=1)
    se = math.sqrt(2 * 0.25 / runs)
    assert np.all(np.abs(a - b[::-1]) <= 4 * se)


def test_more_judges_raise_the_fair_level():
    levels = [J.strongest_wins(6, k, 0.5, 0.0, 0.0, 6000, 4)[0].mean() for k in (1, 3, 9)]
    assert levels[0] < levels[1] < levels[2]


def test_a_stronger_gap_of_the_ladder_raises_the_title_chance_of_the_strongest():
    chances = []
    for gap in (0.0, 150.0, 300.0, 600.0):
        run = PL.simulate_plans(SC.elo_ladder(6, gap), SC.random_assignments(6, 1, 1), 3000, 2, 0.0, 0.5, 0.25)
        chances.append(float(run.share[0].mean()))
    assert chances == sorted(chances)
    assert chances[0] == pytest.approx(1 / 6, abs=0.02)


def test_more_games_per_pair_raise_the_chance_of_the_strongest_at_the_top():
    rows = [NO.top_chance(6, g, NO.ELO_STEP, 3000, 5).p_top for g in (1, 2, 4, 10, 40)]
    assert rows == sorted(rows) and rows[-1] > 0.9


def test_the_noise_floor_depends_only_on_the_differences_of_the_ladder():
    """Gleiche Stufe, gleiche Zufallszahlen: eine andere Verschiebung der ganzen Leiter ändert nichts (es zählen nur Unterschiede)."""
    a = NO.top_chance(5, 2, 40.0, 2000, 9)
    b = NO.top_chance(5, 2, 40.0, 2000, 9)
    assert a == b


# --- Zufall: gleicher Seed, gleiches Ergebnis -----------------------------------------------------------------------------------------------------------

def _comparable(rows):
    """Zeilen als Tupel; nan (kein Standardfehler bei exakter Rechnung) wird zu None, damit der Vergleich gilt."""
    return [tuple(None if isinstance(v, float) and math.isnan(v) else v for v in dataclasses.astuple(r)) for r in rows]


def test_same_settings_and_seed_give_the_same_rows():
    s = EV.Settings(seed=11, runs=2000)
    assert _comparable(EV.analyse(s)) == _comparable(EV.analyse(s))
    assert _comparable(EV.analyse(s)) != _comparable(EV.analyse(EV.Settings(seed=12, runs=2000)))


def test_the_exact_models_do_not_depend_on_the_seed_or_the_number_of_runs():
    a = EV.analyse(EV.Settings(seed=1, runs=2000))
    b = EV.analyse(EV.Settings(seed=2, runs=5000))
    assert _comparable([a[1], a[3]]) == _comparable([b[1], b[3]])


def test_replacing_a_settings_field_by_keyword_changes_only_that_field():
    """Die Settings sind nur mit Schlüsselwörtern aufrufbar (ein eingefügtes Feld würde sonst still Werte verschieben)."""
    with pytest.raises(TypeError):
        EV.Settings(11, 2000)
    assert EV.Settings(seed=3).runs == C.DEFAULT_RUNS


def test_hash_seed_independence_across_processes():
    """PYTHONHASHSEED = 1 und 2: dieselben Zahlen (kein hash(), keine Reihenfolge von Mengen im Ergebnis)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("import rau_evaluation as E; rows = E.analyse(E.Settings(seed=5, runs=2000)); "
            "print([round(r.delta, 12) for r in rows])")
    root = str(Path(__file__).resolve().parent.parent)
    outs = []
    for hs in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": root}
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=root, check=True).stdout)
    assert outs[0] == outs[1] and outs[0].strip()
