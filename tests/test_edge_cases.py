"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 2, ein Lauf, eine Zuordnung, Raten null, Wahrscheinlichkeiten 0 und 1."""

import math

import numpy as np
import pytest

import rau_duel as D
import rau_evaluation as EV
import rau_judges as J
import rau_lage as LG
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_scenario as SC
import rau_shootout as SO


def test_two_teams_are_the_smallest_league():
    """n = 2: eine Zirkelrunde, ein Spiel; die Titelchance der Stärkeren ist die Siegchance plus der halbe Remisanteil (Gleichstand zu gleichen Teilen)."""
    ratings = SC.elo_ladder(2, 200.0)
    run = PL.simulate_plans(ratings, np.array([[0, 1], [1, 0]]), 20000, 3, 0.0, 0.5, 0.25)
    w, d, _ = PL.result_probabilities(100.0, -100.0, 0.25)
    assert run.share[0].mean() == pytest.approx(w + d / 2, abs=0.01)
    assert (run.share[0] == run.share[1]).all()


def test_one_run_and_one_assignment():
    run = PL.simulate_plans(SC.elo_ladder(4, 100.0), np.array([[0, 1, 2, 3]]), 1, 1, 50.0, 0.5, 0.25)
    assert run.share.shape == (1, 1)
    with pytest.raises(ValueError):
        LG.split_best_worst(run.share)


def test_a_single_assignment_has_no_lage_difference():
    ratings = SC.elo_ladder(4, 100.0)
    run = PL.simulate_plans(ratings, np.array([[0, 1, 2, 3]]), 400, 1, 50.0, 0.5, 0.25)
    lage = LG.split_best_worst(run.share, run.points)
    assert lage.best == lage.worst == 0 and lage.delta == 0.0


def test_equal_ladder_and_no_form_gives_the_fair_share_of_the_title():
    run = PL.simulate_plans(SC.elo_ladder(6, 0.0), SC.random_assignments(6, 1, 1), 30000, 2, 0.0, 0.5, 0.25)
    assert run.share[0].mean() == pytest.approx(1 / 6, abs=0.01)


def test_a_single_judge_and_two_performers():
    wins = J.strongest_wins(2, 1, 0.5, 0.0, 0.0, 500, 1)
    assert wins.shape == (2, 500) and 0.5 < wins.mean() < 1.0


def test_one_run_of_a_judging_competition():
    assert J.strongest_wins(3, 2, 0.5, 0.05, 0.0, 1, 1).shape == (3, 1)


def test_duel_with_certain_goals_and_a_large_gap():
    """Eine riesige Stärke der einen Seite: sie kommt fast sicher weiter, Restmasse bleibt klein."""
    res = D.advance_chance(2.5, 2.0, 300.0, True, True, True)
    assert 0.9 < res.p_strong <= 1.0 and abs(res.residual) < 1e-9


def test_duel_probabilities_stay_between_zero_and_one_at_all_corners():
    for lh in (0.5, 2.5):
        for la in (0.3, 2.0):
            for elo in (0.0, 300.0):
                for rule in (False, True):
                    for et in (False, True):
                        for home in (False, True):
                            p = D.advance_chance(lh, la, elo, home, rule, et).p_strong
                            assert 0.0 <= p <= 1.0


def test_shootout_with_probability_one_and_zero():
    assert SO.first_team_win(1.0, 1.0, 0.0, "ABAB", 5) == pytest.approx(0.5)      # alle treffen, Sudden Death nie entschieden: Rückfall 0,5
    assert SO.first_team_win(0.0, 0.0, 0.0, "ABBA", 5) == pytest.approx(0.5)
    assert SO.first_team_win(1.0, 0.0, 0.0, "ABAB", 5) == pytest.approx(1.0)


def test_shootout_pressure_cannot_make_a_probability_negative():
    assert 0.0 <= SO.first_team_win(0.5, 0.3, 0.25, "ABAB", 5) <= 1.0
    assert 0.0 <= SO.first_team_win(0.5, 0.3, 0.25, "ABBA", 5) <= 1.0


def test_power_with_one_competition_and_extreme_chances():
    assert PW.exact_power(0.9, 0.1, 1, 0.05) == 0.0                # bei einer Beobachtung je Lage ist nie ein Unterschied nachweisbar
    assert PW.exact_power(1.0, 0.0, 30, 0.05) == pytest.approx(1.0)
    assert PW.binomial_window(1, 0.5)[0] == 0


def test_certain_chances_need_one_observation_formula_gives_finite_numbers():
    assert math.isfinite(PW.n_two_groups(0.99, 0.01, 0.05, 0.8))
    assert PW.n_two_groups(0.99, 0.01, 0.05, 0.8) < 5


def test_noise_floor_at_the_corners():
    assert NO.top_chance(2, 1, 0.0, 2000, 1).p_top == pytest.approx(0.5, abs=0.03)
    assert NO.top_chance(12, 40, NO.ELO_STEP, 100, 1).p_top > 0.8


def test_negative_or_zero_delta_means_no_number_of_competitions():
    row = EV.Row("Hin- und Rückspiel", "a", "b", 0.5, 0.5, 0.0, float("nan"), True)
    assert math.isinf(EV.needed_for(row, 0.05, 0.8, "zwei Lagen"))
    negative = EV.Row("Auftrittsreihenfolge", "a", "b", 0.5, 0.52, -0.02, 0.01, False)
    assert math.isinf(EV.needed_for(negative, 0.05, 0.8, "zwei Lagen"))
    assert EV.needed(negative, 0.05, 0.8, "zwei Lagen") == EV.Needed(math.inf, math.inf, math.inf)


def test_unknown_model_in_example_row():
    with pytest.raises(ValueError):
        EV.example_row([], "Unbekannt")


def test_exact_lage_with_equal_chances():
    lage = LG.exact_lage([0.5, 0.5])
    assert lage.delta == 0.0 and lage.best == 0 and lage.worst == 0
