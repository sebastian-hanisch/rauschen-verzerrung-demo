"""Einheiten gegen Handrechnung auf Mini-Instanzen (DEMO-PLAYBOOK §1b): jede Regel der Kerne als eigene Funktion."""

import math

import numpy as np
import pytest

import rau_duel as D
import rau_judges as J
import rau_lage as LG
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_scenario as SC
import rau_shootout as SO
from rau_rng import SplitMix64


# --- Szenario: Zufallsblöcke, Elo, Zirkeltafel, Zuordnungen --------------------------------------------------------------------------------------------------

def test_uniform_block_is_bit_equal_to_the_splitmix_class():
    rng = SplitMix64(2026)
    want = [rng.random() for _ in range(50)]
    assert list(SC.uniform_block(2026, 50)) == want
    assert list(SC.uniform_block(2026, 10, start=40)) == want[40:50]


def test_normal_block_has_mean_zero_and_unit_variance_and_matches_the_class_in_distribution():
    z = SC.normal_block(5, 40000)
    assert abs(z.mean()) < 0.03 and abs(z.var() - 1.0) < 0.04
    # die erste Zahl entspricht der Klasse (Box-Muller, derselbe Kosinus-Zweig) bis auf die letzte Binärstelle
    assert SC.normal_block(5, 1)[0] == pytest.approx(SplitMix64(5).normal(), abs=1e-9)


def test_elo_expectation_by_hand():
    assert SC.elo_expectation(0) == pytest.approx(0.5)
    assert SC.elo_expectation(100) == pytest.approx(1 / (1 + 10 ** (-0.25)))   # 0,640...
    assert SC.elo_expectation(100) + SC.elo_expectation(-100) == pytest.approx(1.0)


def test_elo_ladder_by_hand():
    assert list(SC.elo_ladder(5, 400.0)) == [200.0, 100.0, 0.0, -100.0, -200.0]
    assert list(SC.elo_ladder(1, 300.0)) == [0.0]
    assert not SC.elo_ladder(4, 0.0).any()
    with pytest.raises(ValueError):
        SC.elo_ladder(0, 100.0)


def test_circle_rounds_for_four_teams_by_hand():
    """Berger-Tafel n = 4: Platz 0 bleibt, die übrigen rücken weiter."""
    assert SC.circle_rounds(4) == [[(0, 3), (1, 2)], [(0, 2), (3, 1)], [(0, 1), (2, 3)]]


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_circle_rounds_are_a_complete_round_robin(n):
    rounds = SC.circle_rounds(n)
    assert len(rounds) == n - 1
    pairs = [tuple(sorted(p)) for rnd in rounds for p in rnd]
    assert len(pairs) == len(set(pairs)) == n * (n - 1) // 2
    for rnd in rounds:
        assert sorted(x for p in rnd for x in p) == list(range(n))


def test_circle_rounds_reject_odd_or_tiny_n():
    for n in (0, 1, 3, 5):
        with pytest.raises(ValueError):
            SC.circle_rounds(n)


def test_random_assignments_by_hand():
    four = SC.random_assignments(4, 100, 1)
    assert len(four) == 24 and len({tuple(r) for r in four}) == 24       # mehr verlangt als es gibt: alle 4! = 24
    assert list(four[0]) == [0, 1, 2, 3]
    eight = SC.random_assignments(8, 24, 5)
    assert len(eight) == 24 and len({tuple(r) for r in eight}) == 24
    assert all(sorted(r) == list(range(8)) for r in eight)
    assert (SC.random_assignments(8, 24, 5) == eight).all()
    with pytest.raises(ValueError):
        SC.random_assignments(4, 0, 1)


# --- Lage -------------------------------------------------------------------------------------------------------------------------------------------------

def test_exact_lage_picks_the_largest_and_smallest_chance():
    lage = LG.exact_lage([0.55, 0.62, 0.49])
    assert (lage.best, lage.worst) == (1, 2) and lage.delta == pytest.approx(0.13)
    assert math.isnan(lage.se) and lage.n_measure == 0


def test_exact_lage_treats_rounding_residue_as_zero():
    assert LG.exact_lage([0.6, 0.6 + 1e-15]).delta == 0.0


def test_split_best_worst_selects_on_the_first_half_and_measures_on_the_second():
    # Lage 0: Auswahlhälfte hoch (1,1,1,1), Messhälfte (0,0,0,0); Lage 1: umgekehrt. Ausgewählt wird nach der ersten Hälfte, gemessen an der zweiten.
    outcome = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=float)
    lage = LG.split_best_worst(outcome)
    assert (lage.best, lage.worst) == (0, 1)
    assert lage.p_best == 0.0 and lage.p_worst == 1.0 and lage.delta == -1.0 and lage.n_measure == 4


def test_split_best_worst_uses_the_select_array_and_a_paired_standard_error():
    outcome = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=float)
    select = np.array([[0, 0, 0, 0], [5, 5, 0, 0]], dtype=float)
    lage = LG.split_best_worst(outcome, select)
    assert (lage.best, lage.worst) == (1, 0)             # Auswahl nach `select` (erste Hälfte: 5 gegen 0)
    d = np.array([0, 0]) - np.array([1, 0])             # Messhälfte: Lage 1 (0, 0) minus Lage 0 (1, 0)
    assert lage.delta == pytest.approx(d.mean()) and lage.se == pytest.approx(d.std(ddof=1) / math.sqrt(2))


def test_split_best_worst_rejects_a_single_run():
    with pytest.raises(ValueError):
        LG.split_best_worst(np.array([[1.0], [0.0]]))


# --- Spielplan und Form ----------------------------------------------------------------------------------------------------------------------------------

def test_draw_split_by_hand():
    win, draw = PL.draw_split(0.5, 0.25)
    assert (win, draw) == pytest.approx((0.375, 0.25))                  # Remis d = min(0,25, 2·0,5) = 0,25; Sieg 0,5 - 0,125
    win, draw = PL.draw_split(0.9, 0.25)
    assert (win, draw) == pytest.approx((0.9 - 0.1, 0.2))               # d = min(0,25, 2·0,1) = 0,2
    win, draw = PL.draw_split(0.5, 0.0)
    assert (win, draw) == (0.5, 0.0)


def test_result_probabilities_sum_to_one_and_keep_the_expected_score():
    w, d, l = PL.result_probabilities(100.0, 0.0, 0.25)
    assert w + d + l == pytest.approx(1.0)
    assert w + d / 2 == pytest.approx(SC.elo_expectation(100.0))


def test_start_distribution_is_a_fixed_point_and_independent_of_the_memory():
    ratings = np.array([150.0, 50.0, -50.0, -150.0])
    pi = PL.form_start_distribution(ratings, 50.0, 0.5, 0.25)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.allclose(pi, PL.form_start_distribution(ratings, 50.0, 0.0, 0.25), atol=1e-9)
    assert np.allclose(pi, PL.form_start_distribution(ratings, 50.0, 0.9, 0.25), atol=1e-9)
    assert pi[0, 2] > pi[0, 0] and pi[3, 0] > pi[3, 2]                    # der Stärkste ist öfter heiß, der Schwächste öfter kalt
    assert np.allclose(pi[0], pi[3][::-1])                                # symmetrische Leiter


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("m, rho", [(0.0, 0.5), (50.0, 0.5), (150.0, 0.0), (100.0, 0.95)])
def test_start_distribution_is_normalised_for_every_league_size(n, m, rho):
    """Regressionstest (Fehler im Bau): ohne Normierung je Schritt schwand die Masse der Verteilung bei acht Mannschaften auf 1e-13, und alle Mannschaften starteten im Zustand +1."""
    pi = PL.form_start_distribution(SC.elo_ladder(n, 300.0), m, rho, 0.25)
    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-12) and (pi >= 0).all()


def test_start_distribution_without_form_effect_is_the_result_distribution_against_the_field():
    """m = 0: der Zustand ist das letzte Ergebnis, die Verteilung also die Ergebnisverteilung gegen das Feld (Mittel über die anderen)."""
    ratings = np.array([100.0, 0.0, -100.0, -200.0])
    pi = PL.form_start_distribution(ratings, 0.0, 0.5, 0.25)
    for i in range(4):
        w, d, l = PL.result_probabilities(ratings[i], np.delete(ratings, i), 0.25)
        assert pi[i] == pytest.approx([l.mean(), d.mean(), w.mean()], abs=1e-9)


def test_simulate_plans_shapes_and_ranges():
    ratings = SC.elo_ladder(4, 300.0)
    perms = SC.random_assignments(4, 5, 1)
    run = PL.simulate_plans(ratings, perms, 300, 7, 50.0, 0.5, 0.25)
    assert run.share.shape == run.points.shape == (5, 300)
    assert run.points.min() >= 0 and run.points.max() <= 9                # 3 Partien, höchstens 3 Punkte je Partie
    assert run.share.min() >= 0.0 and run.share.max() <= 1.0
    assert (run.share[run.points == 9] == 1.0).all()                      # 9 Punkte aus 3 Siegen: Alleinsieg ist sicher (kein anderer kann 9 haben)


def test_simulation_without_form_gives_the_same_result_for_every_assignment():
    ratings = SC.elo_ladder(6, 300.0)
    perms = SC.random_assignments(6, 12, 3)
    for double in (False, True):
        run = PL.simulate_plans(ratings, perms, 400, 5, 0.0, 0.5, 0.25, double=double)
        assert (run.share == run.share[0]).all() and (run.points == run.points[0]).all()


def test_simulation_with_form_differs_between_assignments():
    ratings = SC.elo_ladder(8, 300.0)
    perms = SC.random_assignments(8, 12, 3)
    run = PL.simulate_plans(ratings, perms, 2000, 5, 100.0, 0.0, 0.25)
    assert not (run.points == run.points[0]).all()


def test_simulation_is_deterministic():
    ratings = SC.elo_ladder(4, 200.0)
    perms = SC.random_assignments(4, 4, 1)
    a = PL.simulate_plans(ratings, perms, 200, 9, 50.0, 0.5, 0.25)
    b = PL.simulate_plans(ratings, perms, 200, 9, 50.0, 0.5, 0.25)
    assert (a.share == b.share).all() and (a.points == b.points).all()


def test_double_round_doubles_the_games():
    ratings = SC.elo_ladder(4, 300.0)
    perms = SC.random_assignments(4, 1, 1)
    run = PL.simulate_plans(ratings, perms, 200, 3, 0.0, 0.5, 0.25, double=True)
    assert run.points.max() <= 18 and run.points.max() > 9


def test_simulate_plans_needs_at_least_one_run():
    with pytest.raises(ValueError):
        PL.simulate_plans(SC.elo_ladder(4, 100.0), np.array([[0, 1, 2, 3]]), 0, 1, 0.0, 0.5, 0.25)


# --- Hin- und Rückspiel ----------------------------------------------------------------------------------------------------------------------------------

def test_poisson_pmf_by_hand():
    p = D.poisson_pmf(1.0)
    assert p[0] == pytest.approx(math.exp(-1)) and p[1] == pytest.approx(math.exp(-1)) and p[2] == pytest.approx(math.exp(-1) / 2)
    assert 1.0 - p.sum() <= D.TAIL * 1.01
    assert list(D.poisson_pmf(0.0)) == [1.0]
    with pytest.raises(ValueError):
        D.poisson_pmf(-0.1)


def test_difference_pmf_is_a_distribution_with_the_right_mean():
    values, weights = D.difference_pmf(1.4, 0.6)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (values * weights).sum() == pytest.approx(0.8, abs=1e-9)


def test_team_factors_by_hand():
    fs, fw = D.team_factors(400.0)
    assert fs == pytest.approx(math.sqrt(10)) and fw == pytest.approx(1 / math.sqrt(10)) and fs / fw == pytest.approx(10.0)
    assert D.team_factors(0.0) == (1.0, 1.0)


def test_no_goals_ever_leaves_everything_to_the_penalties():
    """Torraten null: 0:0 und 0:0, bei Regel gleich viele Auswärtstore (0), bei Verlängerung wieder 0:0, also entscheidet das Elfmeterschießen mit der gegebenen Chance."""
    for rule in (False, True):
        for et in (False, True):
            res = D.advance_chance(0.0, 0.0, 0.0, True, rule, et, 0.7)
            assert res.p_strong == pytest.approx(0.7) and res.p_decided_90 == pytest.approx(0.0)


def test_equal_teams_without_extra_time_and_rule_are_a_coin_flip_in_every_lage():
    for lh, la in ((1.5, 1.1), (0.7, 1.4), (2.0, 0.5)):
        assert D.advance_chance(lh, la, 0.0, True, False, False).p_strong == pytest.approx(0.5, abs=1e-12)
        assert D.advance_chance(lh, la, 0.0, False, False, False).p_strong == pytest.approx(0.5, abs=1e-12)


def test_extra_time_split_by_hand():
    """Ein Drittel der Rate: Raten 0,6 und 0,3 ergeben 0,2 und 0,1 in der Verlängerung; P(0:0) = exp(-0,3)."""
    home, away, equal, zero = D.extra_time_split(0.6, 0.3)
    assert zero == pytest.approx(math.exp(-0.3))
    assert home + away + equal + zero == pytest.approx(1.0, abs=1e-12)
    assert home > away


def test_extra_time_split_equal_scoring_by_hand():
    """P(gleich viele mit mindestens einem Tor) = Summe über k >= 1 von P(k) P(k); bei den Raten 0,2 und 0,1 gilt k = 1: e^-0,3 · 0,2 · 0,1 plus kleine Glieder."""
    home, away, equal, zero = D.extra_time_split(0.6, 0.3)
    k1 = math.exp(-0.3) * 0.2 * 0.1
    k2 = math.exp(-0.3) * (0.2 ** 2 / 2) * (0.1 ** 2 / 2)
    assert equal == pytest.approx(k1 + k2, rel=0.01)


def test_advance_chance_rejects_a_bad_penalty_probability():
    with pytest.raises(ValueError):
        D.advance_chance(1.5, 1.1, 0.0, True, False, True, 1.2)


def test_lage_effect_signs_and_swap():
    good, bad, delta = D.lage_effect(1.5, 1.1, 0.0)
    assert delta == pytest.approx(good - bad) and delta > 0
    good2, bad2, delta2 = D.lage_effect(1.3, 1.3, 100.0, True, True)       # mit Regel und gleichen Raten: das Rückspiel zu Hause ist ein Nachteil
    assert delta2 < 0


# --- Auftrittsreihenfolge --------------------------------------------------------------------------------------------------------------------------------

def test_panel_scores_by_hand():
    """Zwei Auftretende, ein Richter, kein Rauschen: Note = q + β·(k - 0,5) + γ·vorherige Note. Platz 0: 1 + 0,2·(-0,5) = 0,9; Platz 1: 2 + 0,2·0,5 + 0,5·0,9 = 2,55."""
    q = np.array([[1.0, 2.0]])
    e = np.zeros((1, 2, 1))
    out = J.panel_scores(q, e, 0.5, 0.2, 0.5)
    assert out[0] == pytest.approx([0.9, 2.55])


def test_panel_scores_average_the_judges():
    q = np.array([[0.0]])
    e = np.array([[[1.0, 3.0]]])
    assert J.panel_scores(q, e, 0.5, 0.0, 0.0)[0, 0] == pytest.approx(0.5 * 2.0)


def test_order_with_the_strongest_at_a_position_by_hand():
    q = np.array([[0.1, 0.9, 0.5, 0.3]])
    keys = np.array([[0.4, 0.0, 0.1, 0.9]])        # Reihenfolge der Übrigen nach den Schlüsseln: 2 (0,1), 0 (0,4), 3 (0,9)
    assert list(J.order_with_strongest_at(q, keys, 0)[0]) == [1, 2, 0, 3]
    assert list(J.order_with_strongest_at(q, keys, 2)[0]) == [2, 0, 1, 3]
    assert list(J.order_with_strongest_at(q, keys, 3)[0]) == [2, 0, 3, 1]
    with pytest.raises(ValueError):
        J.order_with_strongest_at(q, keys, 4)


def test_strongest_wins_without_noise_is_certain():
    """Ohne Rauschen, Drift und Anker gewinnt die Qualitätsbeste immer."""
    wins = J.strongest_wins(5, 3, 0.0, 0.0, 0.0, 200, 3)
    assert (wins == 1.0).all()


def test_strongest_wins_shape_and_values():
    wins = J.strongest_wins(4, 3, 0.5, 0.05, 0.0, 100, 1)
    assert wins.shape == (4, 100) and set(np.unique(wins)) <= {0.0, 1.0}


def test_strongest_wins_needs_two_performers_and_a_judge():
    with pytest.raises(ValueError):
        J.strongest_wins(1, 3, 0.5, 0.0, 0.0, 10, 1)
    with pytest.raises(ValueError):
        J.strongest_wins(3, 0, 0.5, 0.0, 0.0, 10, 1)


def test_a_large_drift_makes_the_last_place_nearly_certain_and_the_first_nearly_hopeless():
    wins = J.strongest_wins(6, 5, 0.5, 0.6, 0.0, 3000, 2)
    assert wins[-1].mean() > 0.95 and wins[0].mean() < 0.05


# --- Elfmeterschießen ------------------------------------------------------------------------------------------------------------------------------------

def test_one_shot_case_by_hand():
    """R = 1, p = 0,5, d = 0,1: A trifft und B trifft (B im Rückstand 0,4): 0,2; A trifft, B verfehlt: 0,3; A verfehlt, B trifft: 0,25; beide verfehlen: 0,25.
    A gewinnt 0,3 direkt, bei Gleichstand (0,45) den Sudden Death mit 0,3/(0,3 + 0,25) = 6/11: gesamt 6/11."""
    assert SO.first_team_win(0.5, 0.5, 0.1, "ABAB", 1) == pytest.approx(6 / 11)
    dist = SO.regulation_distribution(0.5, 0.5, 0.1, "ABAB", 1)
    assert dist[(1, 0)] == pytest.approx(0.3) and dist[(0, 1)] == pytest.approx(0.25) and dist[(1, 1)] == pytest.approx(0.2) and dist[(0, 0)] == pytest.approx(0.25)


def test_regulation_distribution_is_a_distribution():
    for order in SO.ORDERS:
        assert sum(SO.regulation_distribution(0.7, 0.6, 0.1, order, 5).values()) == pytest.approx(1.0)


def test_sudden_death_closed_form_by_hand():
    """ABAB, p = 0,5, d = 0,1: a1 = 0,5·0,6 = 0,3, b1 = 0,5·0,5 = 0,25, also 0,3/0,55."""
    assert SO.sudden_death_first_wins(0.5, 0.5, 0.1, "ABAB", 1) == pytest.approx(0.3 / 0.55)
    # unabhängige Schüsse (d = 0): Gleichstand im Sudden Death ist eine Münze
    assert SO.sudden_death_first_wins(0.7, 0.7, 0.0, "ABAB", 5) == pytest.approx(0.5)
    assert SO.sudden_death_first_wins(0.7, 0.7, 0.0, "ABBA", 5) == pytest.approx(0.5)


def test_without_pressure_the_order_is_worthless():
    for order in SO.ORDERS:
        good, bad, delta = SO.lage_effect(0.8, 0.1, 0.0, order)
        assert delta == pytest.approx(0.0, abs=1e-12)


def test_pressure_makes_the_first_shooter_better_off_in_abab():
    good, bad, delta = SO.lage_effect(0.75, 0.0, 0.05, "ABAB")
    assert good == pytest.approx(0.5455, abs=1e-4) and bad == pytest.approx(1 - 0.5455, abs=1e-4)
    assert delta == pytest.approx(2 * (good - 0.5))


def test_shootout_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SO.regulation_distribution(0.7, 0.7, 0.0, "AABB", 5)
    with pytest.raises(ValueError):
        SO.regulation_distribution(0.7, 0.7, 0.0, "ABAB", 0)
    with pytest.raises(ValueError):
        SO.lage_effect(0.5, 0.6, 0.0)
    with pytest.raises(ValueError):
        SO.lage_effect(0.3, 0.4, 0.0)


# --- Benötigte Zahl, Test --------------------------------------------------------------------------------------------------------------------------------

def test_two_group_formula_by_hand():
    """p1 = 0,6, p2 = 0,4, α = 5 %, Teststärke 80 %: (1,95996·√0,5 + 0,84162·√0,48)² / 0,04 = 96,93."""
    za, zb = 1.959964, 0.841621
    want = (za * math.sqrt(2 * 0.5 * 0.5) + zb * math.sqrt(0.24 + 0.24)) ** 2 / 0.04
    assert PW.n_two_groups(0.6, 0.4, 0.05, 0.8) == pytest.approx(want, rel=1e-5)
    assert want == pytest.approx(96.93, abs=0.01)
    assert PW.whole_competitions(PW.n_two_groups(0.6, 0.4, 0.05, 0.8)) == 97.0


def test_against_fair_formula_by_hand():
    """p0 = 0,5, p1 = 0,6: ((1,95996·0,5 + 0,84162·√0,24) / 0,1)² = 15,48² ..."""
    za, zb = 1.959964, 0.841621
    want = ((za * 0.5 + zb * math.sqrt(0.24)) / 0.1) ** 2
    assert PW.n_against_fair(0.5, 0.6, 0.05, 0.8) == pytest.approx(want, rel=1e-5)


def test_n_needed_dispatch_and_unknown_design():
    assert PW.n_needed(0.6, 0.4, 0.05, 0.8) == PW.n_two_groups(0.6, 0.4, 0.05, 0.8)
    assert PW.n_needed(0.6, 0.4, 0.05, 0.8, "gegen den Fairwert") == PW.n_against_fair(0.5, 0.6, 0.05, 0.8)
    with pytest.raises(ValueError):
        PW.n_needed(0.6, 0.4, 0.05, 0.8, "unbekannt")


def test_no_difference_needs_infinitely_many():
    assert math.isinf(PW.n_two_groups(0.5, 0.5, 0.05, 0.8))
    assert math.isinf(PW.n_against_fair(0.5, 0.5, 0.05, 0.8))
    assert math.isinf(PW.whole_competitions(math.inf))


def test_levels_must_be_probabilities():
    with pytest.raises(ValueError):
        PW.n_two_groups(0.6, 0.4, 0.0, 0.8)
    with pytest.raises(ValueError):
        PW.n_two_groups(0.6, 0.4, 0.05, 1.0)
    with pytest.raises(ValueError):
        PW.z_quantile(1.0)


def test_binomial_window_by_hand():
    lo, w = PW.binomial_window(2, 0.5)
    assert lo == 0 and list(w) == pytest.approx([0.25, 0.5, 0.25])
    lo, w = PW.binomial_window(5, 0.0)
    assert lo == 0 and list(w) == [1.0]
    lo, w = PW.binomial_window(5, 1.0)
    assert lo == 5 and list(w) == [1.0]


def test_rejects_two_groups_by_hand():
    """40 gegen 60 von je 100: p̂ = 0,5, z = 0,2/√(2·0,25/100) = 2,83 ≥ 1,96 (lehnt ab); 48 gegen 52: z = 0,566 (lehnt nicht ab); 0 gegen 0: kein Test."""
    assert PW.rejects_two_groups(40, 60, 100, 0.05)
    assert not PW.rejects_two_groups(48, 52, 100, 0.05)
    assert not PW.rejects_two_groups(0, 0, 100, 0.05)
    assert not PW.rejects_two_groups(100, 100, 100, 0.05)


def test_rejects_against_fair_by_hand():
    """p0 = 0,5, n = 100: Schwelle 1,96 · 0,05 = 0,098; 60 liegt bei 0,1 (lehnt ab), 59 bei 0,09 (nicht)."""
    assert PW.rejects_against_fair(60, 100, 0.5, 0.05)
    assert not PW.rejects_against_fair(59, 100, 0.5, 0.05)


def test_simulate_power_is_deterministic_and_validates():
    a = PW.simulate_power(0.6, 0.5, 100, 0.05, 500, 4)
    b = PW.simulate_power(0.6, 0.5, 100, 0.05, 500, 4)
    assert a == b
    with pytest.raises(ValueError):
        PW.simulate_power(0.6, 0.5, 100, 0.05, 0, 4)
    with pytest.raises(ValueError):
        PW.simulate_power(0.6, 0.5, 100, 0.05, 10, 4, "unbekannt")


def test_season_path_by_hand_with_certain_outcomes():
    """p_good = 1, p_bad = 0: nach s Saisons ist der Unterschied 1, das Band bei gemeinsamem Anteil 0,5 gleich 1,96·√(2·0,25/s); ab s = 2 liegt 1 außerhalb (Band 1,386 bei s = 2: noch nicht)."""
    path = PW.season_path(1.0, 0.0, 10, 0.05, 1)
    assert list(path.diff) == [1.0] * 10
    assert path.band[1] == pytest.approx(1.959964 * math.sqrt(0.5 / 2), rel=1e-5)
    assert path.first_seen == 2 and path.stays_out == 2


def test_season_path_without_difference_stays_in_the_band_more_often_than_not():
    path = PW.season_path(0.5, 0.5, 400, 0.05, 3)
    assert (np.abs(path.diff) < path.band).mean() > 0.8
    with pytest.raises(ValueError):
        PW.season_path(0.5, 0.5, 1, 0.05, 3)


# --- Rauschgrenze ----------------------------------------------------------------------------------------------------------------------------------------

def test_elo_step_equals_the_logistic_step():
    assert NO.ELO_STEP == pytest.approx(0.3 * 400 / math.log(10))
    assert SC.elo_expectation(NO.ELO_STEP) == pytest.approx(1 / (1 + math.exp(-0.3)))


def test_two_teams_one_game_without_draws_is_the_win_probability():
    for step in (50.0, 200.0):
        res = NO.top_chance(2, 1, step, 60000, 5)
        assert abs(res.p_top - SC.elo_expectation(step)) <= 4 * res.se


def test_noise_curve_uses_one_stream_per_point():
    rows = NO.noise_curve(4, (1, 1), 50.0, 300, 3)
    assert rows[0].p_top != rows[1].p_top


def test_top_chance_validates_inputs():
    for args in ((1, 1, 10.0, 10, 1), (3, 0, 10.0, 10, 1), (3, 1, 10.0, 0, 1)):
        with pytest.raises(ValueError):
            NO.top_chance(*args)
