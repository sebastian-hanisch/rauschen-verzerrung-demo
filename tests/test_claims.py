"""Die Zahlen der README (Abschnitte „Ergebnis in Kürze“, „Befunde“, „Befunde und Korrekturen gegenüber dem Plan“, „Gegenprobe“) werden hier nachgerechnet, teils mit Band.

Exakt gerechnete Zahlen (Hin- und Rückspiel, Elfmeterschießen, Formel für N, Teststärke) stehen mit Toleranz der letzten angegebenen Stelle; Simulationen mit festem Zufalls-Seed als Band
(plattformfest sind die Zufallszahlen, nicht die letzte Binärstelle der numpy-Funktionen). Zahlen aus mehreren Zufalls-Seeds stehen in `data/messreihe.json`."""

import math

import numpy as np
import pytest

import rau_constants as C
import rau_duel as D
import rau_evaluation as EV
import rau_judges as J
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_shootout as SO
from rau_scenario import elo_ladder, random_assignments

S = EV.Settings()                                  # Seed 35, 8.000 Läufe, Fehlerniveau 5 %, Teststärke 80 %, zwei Lagen


@pytest.fixture(scope="module")
def default_rows():
    return {r.model: r for r in EV.analyse(S)}


@pytest.fixture(scope="module")
def default_needed(default_rows):
    return {m: EV.needed(r, S.alpha_p, S.power_p, S.design) for m, r in default_rows.items()}


# --- Ergebnis in Kürze: die vier Modelle bei den Voreinstellungen ------------------------------------------------------------------------------------------

def test_default_rows_of_the_exact_models(default_rows):
    duel = default_rows["Hin- und Rückspiel"]
    assert (duel.good, duel.bad) == ("Rückspiel zu Hause", "Hinspiel zu Hause")
    assert 100 * duel.p_good == pytest.approx(75.89, abs=0.005) and 100 * duel.p_bad == pytest.approx(74.55, abs=0.005) and 100 * duel.delta == pytest.approx(1.35, abs=0.005)
    shoot = default_rows["Elfmeterschießen"]
    assert (shoot.good, shoot.bad) == ("schießt zuerst", "schießt als zweite")
    assert 100 * shoot.p_good == pytest.approx(62.18, abs=0.005) and 100 * shoot.p_bad == pytest.approx(53.99, abs=0.005) and 100 * shoot.delta == pytest.approx(8.19, abs=0.005)


def test_default_rows_of_the_simulated_models(default_rows):
    plan = default_rows["Spielplan und Form"]
    assert 100 * plan.delta == pytest.approx(4.61, abs=0.15) and 100 * plan.se == pytest.approx(0.54, abs=0.1)
    assert 100 * plan.p_good == pytest.approx(44.99, abs=0.3) and 100 * plan.p_bad == pytest.approx(40.38, abs=0.3)
    auf = default_rows["Auftrittsreihenfolge"]
    assert (auf.good, auf.bad) == ("Platz 10", "Platz 1")
    assert 100 * auf.delta == pytest.approx(31.37, abs=0.5) and 100 * auf.se == pytest.approx(0.73, abs=0.15)
    assert 100 * auf.p_good == pytest.approx(92.4, abs=0.5) and 100 * auf.p_bad == pytest.approx(61.0, abs=0.7)


def test_needed_numbers_at_the_defaults(default_needed):
    assert default_needed["Hin- und Rückspiel"].n == 16157 and default_needed["Elfmeterschießen"].n == 569
    assert default_needed["Auftrittsreihenfolge"].n == pytest.approx(28, abs=3)
    plan = default_needed["Spielplan und Form"]
    assert plan.n == pytest.approx(1810, rel=0.12) and plan.low == pytest.approx(1184, rel=0.15) and plan.high == pytest.approx(3103, rel=0.2)
    ratio = default_needed["Hin- und Rückspiel"].n / default_needed["Auftrittsreihenfolge"].n
    assert 480 < ratio < 640                                                # README: rund 577 (16.157 / 28)


def test_the_order_of_the_four_models_at_the_defaults(default_needed):
    """Gemessene Reihenfolge der benötigten Zahl (nicht vorab festgelegt): Auftritt < Elfmeter < Spielplan < Rückspiel."""
    order = sorted(default_needed, key=lambda m: default_needed[m].n)
    assert order == ["Auftrittsreihenfolge", "Elfmeterschießen", "Spielplan und Form", "Hin- und Rückspiel"]


def test_against_the_fair_value_needs_about_twice_as_many(default_rows):
    n = {m: EV.needed_for(r, 0.05, 0.8, "gegen den Fairwert") for m, r in default_rows.items()}
    assert n["Hin- und Rückspiel"] == 32136 and n["Elfmeterschießen"] == 1128
    assert n["Spielplan und Form"] == pytest.approx(3634, rel=0.12) and n["Auftrittsreihenfolge"] == pytest.approx(45, abs=4)
    for model, row in default_rows.items():
        two = EV.needed_for(row, 0.05, 0.8, "zwei Lagen")
        assert 1.55 < n[model] / two < 2.05, model                          # README: Verhältnis 1,6 bis 2,0 (gegen den Fairwert durch zwei Lagen)


def test_against_the_fair_value_saves_nothing_over_both_lagen_together(default_rows):
    """README: Über beide Lagen zusammen ist die Zahl etwa gleich (zwei Lagen: 2 · N; beim Auftritt 56 gegen 45). Der Fairwert zu kennen spart also keine Wettbewerbe."""
    for model, row in default_rows.items():
        both = 2 * EV.needed_for(row, 0.05, 0.8, "zwei Lagen")
        fair = EV.needed_for(row, 0.05, 0.8, "gegen den Fairwert")
        assert 0.75 < fair / both < 1.05, model
    auf = default_rows["Auftrittsreihenfolge"]
    assert (2 * EV.needed_for(auf, 0.05, 0.8, "zwei Lagen"), EV.needed_for(auf, 0.05, 0.8, "gegen den Fairwert")) == (56, 45)


def test_the_fair_value_is_the_middle_of_the_two_lagen_not_the_chance_without_distortion(default_rows):
    """README: Fairwert = Mitte der beiden Lagen; beim Auftritt 76,7 % gegen 81,7 % ohne Drift, beim Spielplan 42,7 % gegen das Fairniveau 42,0 %."""
    auf = default_rows["Auftrittsreihenfolge"]
    none = J.strongest_wins(10, 5, 0.5, 0.0, 0.0, 20000, 35).mean()
    assert 100 * auf.p_fair == pytest.approx(76.7, abs=0.5) and 100 * none == pytest.approx(81.7, abs=0.6)
    assert 4.0 < 100 * (none - auf.p_fair) < 6.0                              # README: rund 5 Pp
    plan = default_rows["Spielplan und Form"]
    assert 100 * plan.p_fair == pytest.approx(42.7, abs=0.4)
    assert 100 * EV.plan_row(8, 300, 20000, 101, 0, 0.5, False).p_good == pytest.approx(42.0, abs=0.4)


def test_the_square_of_the_delta_ratio_is_near_the_ratio_of_the_needed_numbers(default_rows):
    """README: das Verhältnis der Δ (31,37 gegen 1,35 Pp) ergibt quadriert rund 540, gemessen ist 577."""
    squared = (default_rows["Auftrittsreihenfolge"].delta / default_rows["Hin- und Rückspiel"].delta) ** 2
    assert 500 < squared < 580


def test_the_plan_row_of_the_table_is_a_single_sample():
    """README: bei 8.000 Läufen und den Zufalls-Seeds 101 bis 105 liegt Δ des Spielplans zwischen 2,01 und 4,46 Pp (Mittel 3,34, Streuung 0,98 Pp, angezeigter Standardfehler 0,54 Pp), N zwischen 1.922 und 9.485."""
    rows = [EV.plan_row(8, 300, 8000, seed, 50, 0.5, False) for seed in (101, 102, 103, 104, 105)]
    deltas = [100 * r.delta for r in rows]
    assert min(deltas) == pytest.approx(2.01, abs=0.1) and max(deltas) == pytest.approx(4.46, abs=0.1)
    assert float(np.mean(deltas)) == pytest.approx(3.34, abs=0.1) and float(np.std(deltas, ddof=1)) == pytest.approx(0.98, abs=0.1)
    ns = [EV.needed_for(r, 0.05, 0.8, "zwei Lagen") for r in rows]
    assert min(ns) == pytest.approx(1922, rel=0.05) and max(ns) == pytest.approx(9485, rel=0.05)
    assert float(np.std(deltas, ddof=1)) > 1.5 * float(np.mean([100 * r.se for r in rows]))      # Streuung zwischen den Seeds größer als der Standardfehler


def test_exact_power_at_the_formula_numbers(default_rows):
    """Die exakte Teststärke bei der N der Formel liegt bei den vier Modellen zwischen 79,9 und 83,0 % (zwei Lagen)."""
    powers = {}
    for model, row in default_rows.items():
        n = int(EV.needed_for(row, 0.05, 0.8, "zwei Lagen"))
        powers[model] = PW.exact_power(row.p_good, row.p_bad, n, 0.05)
    assert powers["Hin- und Rückspiel"] == pytest.approx(0.8000, abs=0.0005) and powers["Elfmeterschießen"] == pytest.approx(0.8026, abs=0.0005)
    assert powers["Spielplan und Form"] == pytest.approx(0.7987, abs=0.01) and powers["Auftrittsreihenfolge"] == pytest.approx(0.829, abs=0.01)
    assert all(0.79 < p < 0.84 for p in powers.values())


def test_the_normal_approximation_is_weak_for_the_judging_model_against_the_fair_value(default_rows):
    """README (Grenzen): Auftritt, gegen den Fairwert: N = 45, exakte Teststärke 74,5 % statt 80 % (Fairwert-Chancen 92 % und 77 %: die Näherung ist am Rand der Chance schlecht)."""
    row = default_rows["Auftrittsreihenfolge"]
    n = int(EV.needed_for(row, 0.05, 0.8, "gegen den Fairwert"))
    assert n == pytest.approx(45, abs=4)
    assert PW.exact_power(row.p_good, row.p_bad, n, 0.05, "gegen den Fairwert") == pytest.approx(0.745, abs=0.03)


def test_the_normal_approximation_is_weak_at_small_n_and_a_fair_value_near_one():
    """README: gegen den Fairwert 0,9 mit Δ = 15 Pp, Fehlerniveau 10 % und Ziel 70 % gibt die Formel N = 59, die exakte Teststärke ist 81,7 %."""
    p_good, p_bad = 0.9 + 0.075, 0.9 - 0.075
    n = PW.whole_competitions(PW.n_needed(p_good, p_bad, 0.10, 0.70, "gegen den Fairwert"))
    assert n == 59
    assert PW.exact_power(p_good, p_bad, int(n), 0.10, "gegen den Fairwert") == pytest.approx(0.817, abs=0.002)


def test_the_probe_of_the_app_at_seed_35(default_rows):
    row = default_rows["Hin- und Rückspiel"]
    sim = PW.simulate_power(row.p_good, row.p_bad, 16157, 0.05, C.POWER_REPS, 35)
    assert sim.rate == pytest.approx(0.790, abs=0.03) and abs(sim.rate - 0.8) <= 3 * sim.se


# --- Hin- und Rückspiel: Gegenprobe zur Nachbar-Demo -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lam_home, lam_away, without_rule, with_rule", [(1.3, 1.3, 0.5000, 0.4971), (1.5, 1.1, 0.5084, 0.5005), (1.7, 0.9, 0.5168, 0.5042)])
def test_duel_equal_teams_reproduce_the_neighbour_values(lam_home, lam_away, without_rule, with_rule):
    """Gegenprobe: gleich starke Teams, mit Verlängerung (Zahlen der README der Nachbar-Demo rueckspiel-demo, gelesen 2026-11-18): Chance des Rückspiel-Gastgebers."""
    assert D.advance_chance(lam_home, lam_away, 0.0, True, False, True).p_strong == pytest.approx(without_rule, abs=5e-5)
    assert D.advance_chance(lam_home, lam_away, 0.0, True, True, True).p_strong == pytest.approx(with_rule, abs=5e-5)


def test_duel_values_of_the_second_leg_without_and_with_the_rule():
    assert [round(100 * D.lage_effect(lh, la, 0.0, False, True)[2], 2) for lh, la in ((1.3, 1.3), (1.5, 1.1), (1.7, 0.9))] == [0.0, 1.68, 3.35]
    assert [round(100 * D.lage_effect(lh, la, 0.0, True, True)[2], 2) for lh, la in ((1.3, 1.3), (1.5, 1.1), (1.7, 0.9))] == [-0.58, 0.09, 0.83]


def test_rule_makes_the_lage_difference_small_and_the_number_huge():
    """Mit Auswärtstorregel und Elo-Abstand 100: bei Heim 1,5 und Auswärts 1,1 ist Δ 0,09 Pp (N 4.011.611); bei Heim 1,3 und Auswärts 1,3 ist es 0,45 Pp mit der anderen Lage als der besseren."""
    row = EV.duel_row(1.5, 1.1, 100, True, True)
    assert 100 * row.delta == pytest.approx(0.09, abs=0.005) and EV.needed_for(row, 0.05, 0.8, "zwei Lagen") == pytest.approx(4011611, rel=0.01)
    flat = EV.duel_row(1.3, 1.3, 100, True, True)
    assert 100 * flat.delta == pytest.approx(0.45, abs=0.005) and flat.good == "Hinspiel zu Hause"


def test_the_difference_with_the_rule_changes_its_sign_near_a_home_mean_of_one_point_four_seven():
    """README: mit Regel wechselt Δ (Rückspiel zu Hause minus Hinspiel zu Hause) bei Heim 1,47 das Vorzeichen; darum ist die Zahl 4.011.611 bei Heim 1,5 nicht stabil."""
    signed = {lh: D.lage_effect(lh, round(2.6 - lh, 10), 100.0, True, True, C.DUEL_PEN_STRONG)[2] for lh in (1.45, 1.46, 1.47, 1.5)}
    assert signed[1.45] < 0 < signed[1.5] and signed[1.46] < 0 < signed[1.47]
    assert 100 * signed[1.47] == pytest.approx(0.004, abs=0.003)


# --- Elfmeterschießen ---------------------------------------------------------------------------------------------------------------------------------------

def test_shootout_abba_shrinks_the_lage_difference_and_raises_the_number():
    abab = EV.shoot_row(0.75, 0.05, 0.05, "ABAB")
    abba = EV.shoot_row(0.75, 0.05, 0.05, "ABBA")
    assert 100 * abab.delta == pytest.approx(8.19, abs=0.005) and 100 * abba.delta == pytest.approx(1.32, abs=0.005)
    n_abab = EV.needed_for(abab, 0.05, 0.8, "zwei Lagen")
    n_abba = EV.needed_for(abba, 0.05, 0.8, "zwei Lagen")
    assert (n_abab, n_abba) == (569, 22016) and n_abba / n_abab == pytest.approx(38.7, abs=0.1)


def test_shootout_equal_teams_reproduce_the_neighbour_values():
    """Gegenprobe (README der Nachbar-Demo reihenfolge-demo, gelesen 2026-11-18): p = 0,75, 5 Schüsse; die erste Mannschaft gewinnt bei d = 0,05 zu 54,5 % (ABAB) und 50,7 % (ABBA)."""
    assert round(100 * SO.first_team_win(0.75, 0.75, 0.05, "ABAB", 5), 1) == 54.5
    assert round(100 * SO.first_team_win(0.75, 0.75, 0.05, "ABBA", 5), 1) == 50.7


# --- Auftrittsreihenfolge: Gegenprobe zur Nachbar-Demo ------------------------------------------------------------------------------------------------------

def test_judging_reproduces_the_neighbour_values():
    """Gegenprobe (README der Nachbar-Demo auftrittsreihenfolge-demo, gelesen 2026-11-18): 10 Auftretende, 5 Kampfrichter, Rauschen 0,5, Drift +0,05, Zufallsreihenfolge, 20.000 Wettbewerbe:
    Qualitätsbeste gewinnt auf Platz 1 in 61,1 % und auf Platz 10 in 91,9 % (dort ab 960 Wettbewerben je Platz), im Mittel über die Plätze in 78,4 %, ohne Verzerrung in 82,1 %."""
    wins = J.strongest_wins(10, 5, 0.5, 0.05, 0.0, 20000, 35)
    assert 100 * wins[0].mean() == pytest.approx(61.1, abs=1.5) and 100 * wins[9].mean() == pytest.approx(91.9, abs=1.5)
    assert 100 * wins.mean() == pytest.approx(78.4, abs=1.0)
    none = J.strongest_wins(10, 5, 0.5, 0.0, 0.0, 20000, 35)
    assert 100 * none.mean() == pytest.approx(82.1, abs=1.0)


def test_judging_chance_rises_with_every_place_at_the_defaults():
    per_place = J.strongest_wins(10, 5, 0.5, 0.05, 0.0, 20000, 35).mean(axis=1)
    assert np.all(np.diff(per_place) > 0)
    assert [round(100 * x, 1) for x in per_place[[0, 4, 9]]] == pytest.approx([61.2, 77.1, 92.5], abs=1.2)


def test_the_fair_level_of_the_judging_model():
    none = J.strongest_wins(10, 5, 0.5, 0.0, 0.0, 20000, 35)
    assert 100 * none.mean() == pytest.approx(81.65, abs=0.6)


# --- Spielplan und Form -------------------------------------------------------------------------------------------------------------------------------------

def test_plan_gegenprobe_to_the_neighbour_demo():
    """Gegenprobe (README der Nachbar-Demo spielplan-form-demo, gelesen 2026-11-18): 8 Mannschaften, Elo-Abstand 300, m = 50, ρ = 0,5, Remisanteil höchstens 0,25: mittlere erwartete Punkte des
    Stärksten 14,63 und Spanne über 100 Zuordnungen 0,43 (dort exakt gerechnet). Hier Simulation mit 40.000 Läufen: Mittel 14,6 und Spanne der Mittelwerte 0,53 (überwiegend die anderen Zuordnungen, nicht das Simulationsrauschen; siehe der nächste Test)."""
    ratings = elo_ladder(8, 300.0)
    run = PL.simulate_plans(ratings, random_assignments(8, 100, 2027), 40000, 2026, 50.0, 0.5, 0.25)
    assert run.points.mean() == pytest.approx(14.6, abs=0.1)
    assert PL.points_spread(run) == pytest.approx(0.53, abs=0.12)


def test_the_gap_to_the_neighbour_spread_comes_from_the_drawn_assignments_not_from_the_simulation_noise():
    """README: bei fünf Ziehungen von je 100 Zuordnungen liegt die Spanne zwischen 0,43 und 0,53, bei drei Zufalls-Seeds der Simulation auf denselben 100 Zuordnungen zwischen 0,53 und 0,55."""
    ratings = elo_ladder(8, 300.0)
    by_draw = [PL.points_spread(PL.simulate_plans(ratings, random_assignments(8, 100, draw), 40000, 2026, 50.0, 0.5, 0.25)) for draw in (2026, 2027, 2028, 2029, 2030)]
    assert min(by_draw) == pytest.approx(0.43, abs=0.04) and max(by_draw) == pytest.approx(0.53, abs=0.04)
    by_seed = [PL.points_spread(PL.simulate_plans(ratings, random_assignments(8, 100, 2027), 40000, seed, 50.0, 0.5, 0.25)) for seed in (2026, 1, 2)]
    assert min(by_seed) == pytest.approx(0.53, abs=0.04) and max(by_seed) == pytest.approx(0.55, abs=0.04)


def test_plan_fair_level_and_form_ladder_from_the_data_file():
    """Fairwert (m = 0): die Titelchance der Stärksten ist in allen 24 Zuordnungen gleich, 42,0 %. Mittel über fünf Zufalls-Seeds mit je 20.000 Läufen (data/messreihe.json), ρ = 0,5; Streuung der fünf Werte bei m = 50: 0,68 Pp."""
    one = EV.plan_row(8, 300, 20000, 101, 0, 0.5, False)
    assert one.delta == 0.0 and 100 * one.p_good == pytest.approx(42.0, abs=0.3)
    sweep = {r["x"]: r for r in EV.sweep_plan(0.5)}
    assert [round(100 * sweep[m]["delta"], 2) for m in (0, 25, 50, 75, 100, 150)] == pytest.approx([0.0, 1.86, 4.12, 5.97, 7.87, 11.29], abs=0.01)
    assert 100 * sweep[50]["spread"] == pytest.approx(0.68, abs=0.01)


def test_plan_memory_lowers_the_lage_difference_in_the_data_file():
    cells = {rho: {r["x"]: r for r in EV.sweep_plan(rho)} for rho in C.SWEEP_FORM_RHO}
    assert [round(100 * cells[rho][50]["delta"], 2) for rho in (0.0, 0.5, 0.8)] == pytest.approx([6.59, 4.12, 1.79], abs=0.01)
    for m in (25, 50, 75, 100, 150):
        assert cells[0.0][m]["delta"] > cells[0.5][m]["delta"] > cells[0.8][m]["delta"] > 0
    for rho in (0.0, 0.5, 0.8):
        ds = [cells[rho][m]["delta"] for m in C.SWEEP_FORM_M]
        assert ds == sorted(ds)


def test_plan_needed_numbers_along_the_form_ladder():
    sweep = EV.sweep_plan(0.5)
    ns = EV.sweep_needed(sweep, 0.05, 0.8, "zwei Lagen")
    assert math.isinf(ns[0]) and [int(n) for n in ns[1:]] == [11059, 2249, 1072, 617, 298]
    ns0 = EV.sweep_needed(EV.sweep_plan(0.0), 0.05, 0.8, "zwei Lagen")
    ns8 = EV.sweep_needed(EV.sweep_plan(0.8), 0.05, 0.8, "zwei Lagen")
    assert [int(n) for n in ns0[1:]] == [4097, 879, 343, 191, 80] and [int(n) for n in ns8[1:]] == [50174, 11990, 5867, 3377, 1723]


def test_data_file_cell_is_reproducible():
    """Die vorgerechnete Zelle (m = 50, ρ = 0,5, Zufalls-Seed 101) wird neu gerechnet: gleiche Zahl (Band für Plattformunterschiede)."""
    cell = EV.load_messreihe()["plan"]["m=50|rho=0.5"]
    k = cell["seed"].index(101)
    row = EV.plan_row(C.DEFAULT_PLAN_N, C.DEFAULT_PLAN_GAP, C.SWEEP_RUNS, 101, 50, 0.5, False)
    assert row.delta == pytest.approx(cell["delta"][k], abs=0.002) and row.p_good == pytest.approx(cell["p_good"][k], abs=0.002)
    assert cell["delta"][k] == pytest.approx(0.052342, abs=1e-6)


def test_the_choice_of_the_best_and_worst_of_many_plans_is_a_selection_not_a_constant():
    """Δ hängt an der Zahl der gezogenen Zuordnungen (m = 50, ρ = 0,5, 10.000 Läufe, Mittel über die Zufalls-Seeds 101 bis 103): 2 Zuordnungen 0,51, 6 2,46, 24 3,51, 100 4,84 Pp."""
    means = {}
    for k in (2, 6, 24, 100):
        means[k] = 100 * float(np.mean([PL.plan_lage(8, 300.0, 10000, seed, 50.0, 0.5, 0.25, k).delta for seed in (101, 102, 103)]))
    assert means[2] == pytest.approx(0.51, abs=0.6) and means[6] == pytest.approx(2.46, abs=0.7) and means[24] == pytest.approx(3.51, abs=0.7) and means[100] == pytest.approx(4.84, abs=0.7)
    assert means[2] < means[6] < means[24] < means[100]


# --- Auftrittsreihenfolge: Messreihe -----------------------------------------------------------------------------------------------------------------------

def test_judging_sweep_values_from_the_data_file():
    drift = {r["x"]: r for r in EV.sweep_judges_drift()}
    assert [round(100 * drift[b]["delta"], 2) for b in C.SWEEP_DRIFT] == pytest.approx([0.0, 15.89, 31.12, 45.08, 56.98, 67.23, 75.53], abs=0.01)
    ns = EV.sweep_needed(EV.sweep_judges_drift(), 0.05, 0.8, "zwei Lagen")
    assert [int(n) for n in ns[1:]] == [97, 28, 15, 10, 7, 6]
    anchor = {r["x"]: r for r in EV.sweep_judges_anchor()}
    assert [round(100 * anchor[g]["delta"], 2) for g in C.SWEEP_ANCHOR] == pytest.approx([28.84, 28.48, 35.17, 42.39], abs=0.01)
    assert [int(n) for n in EV.sweep_needed(EV.sweep_judges_anchor(), 0.05, 0.8, "zwei Lagen")] == [46, 42, 24, 20]
    # README: die Ankerreihe läuft bei Drift 0,05; schon ohne Anker ist Δ dort 31,12 Pp
    assert 100 * drift[0.05]["delta"] == pytest.approx(31.12, abs=0.01)


def test_the_anchor_alone_without_drift_is_smaller_than_in_the_anchor_series():
    """README: der Anker allein (Drift 0) ergibt bei 20.000 Läufen und den Zufalls-Seeds 101 bis 103 im Mittel 9,1 / 1,6 / 5,4 / 14,8 Pp für γ = −0,6 / −0,3 / +0,3 / +0,6 (etwa 2 bis 15 Pp)."""
    means = [100 * float(np.mean([EV.judge_row(10, 5, 0.0, g, 20000, seed).delta for seed in (101, 102, 103)])) for g in C.SWEEP_ANCHOR]
    assert means == pytest.approx([9.1, 1.6, 5.4, 14.8], abs=0.8)
    series = {r["x"]: 100 * r["delta"] for r in EV.sweep_judges_anchor()}
    assert all(m < series[g] for m, g in zip(means, C.SWEEP_ANCHOR))


def test_judging_data_file_cell_is_reproducible():
    cell = EV.load_messreihe()["auftritt_drift"]["0.05"]
    k = cell["seed"].index(101)
    row = EV.judge_row(C.DEFAULT_PERF_N, C.DEFAULT_JUDGES, 0.05, 0.0, C.SWEEP_RUNS, 101)
    assert row.delta == pytest.approx(cell["delta"][k], abs=0.002)
    assert 100 * float(np.std(cell["delta"], ddof=1)) == pytest.approx(0.49, abs=0.02)


# --- Hin- und Rückspiel und Elfmeterschießen: Messreihen -----------------------------------------------------------------------------------------------------

def test_duel_sweep_over_the_home_mean():
    rows = EV.sweep_duel(False)
    assert [round(100 * r["delta"], 2) for r in rows] == [0.0, 0.67, 1.35, 2.02, 2.68, 3.35, 4.01, 4.66]
    ns = EV.sweep_needed(rows, 0.05, 0.8, "zwei Lagen")
    assert math.isinf(ns[0]) and [int(n) for n in ns[1:]] == [64522, 16157, 7200, 4066, 2615, 1827, 1351]


def test_duel_sweep_with_the_rule_has_a_minimum_near_one_point_five():
    rows = EV.sweep_duel(True)
    deltas = [100 * r["delta"] for r in rows]
    assert [round(x, 2) for x in deltas] == [0.45, 0.18, 0.09, 0.37, 0.68, 1.02, 1.41, 1.86]
    assert int(np.argmin(deltas)) == 2


def test_shootout_sweep_over_the_pressure():
    rows = EV.sweep_shoot("ABAB")
    assert [round(100 * r["delta"], 2) for r in rows] == [0.0, 4.27, 8.19, 11.8, 15.14, 21.11, 26.3, 30.86]
    assert [int(n) for n in EV.sweep_needed(rows, 0.05, 0.8, "zwei Lagen")[1:]] == [2091, 569, 274, 166, 85, 55, 40]
    abba = EV.sweep_shoot("ABBA")
    assert [round(100 * r["delta"], 2) for r in abba] == [0.0, 0.62, 1.32, 2.08, 2.92, 4.81, 7.01, 9.52]
    assert all(a["delta"] > b["delta"] for a, b in zip(rows[1:], abba[1:]))


# --- Ständiges Hinsehen ------------------------------------------------------------------------------------------------------------------------------------

def test_looking_every_season_inflates_the_false_alarm_rate():
    """400 gewürfelte Wege ohne jeden Unterschied (Zufalls-Seed 35, Fehlerniveau 5 %): in 3.000 Saisons verlässt die Linie das Band mindestens einmal bei p = 0,5 / 0,75 / 0,9 in 61,8 / 58,5 / 50,8 % der Wege."""
    rates = [PW.peeking_rate(p, 3000, 0.05, 400, 35) for p in (0.5, 0.75, 0.9)]
    assert [round(100 * r, 1) for r in rates] == pytest.approx([61.8, 58.5, 50.8], abs=4.0)
    assert PW.peeking_rate(0.75, 60, 0.05, 400, 35) == pytest.approx(0.3175, abs=0.05)
    assert all(r > 0.3 for r in rates)


def test_looking_every_season_depends_on_the_zufalls_seed_of_the_400_paths():
    """README: mit den Zufalls-Seeds 1 bis 4 liegt der Anteil bei 75 % in 3.000 Saisons zwischen 54 und 62 %, in 60 Saisons zwischen 27 und 29 % (Stichprobenfehler rund 2,5 Pp)."""
    long = [100 * PW.peeking_rate(0.75, 3000, 0.05, 400, seed) for seed in (1, 2, 3, 4)]
    short = [100 * PW.peeking_rate(0.75, 60, 0.05, 400, seed) for seed in (1, 2, 3, 4)]
    assert min(long) == pytest.approx(54, abs=1.0) and max(long) == pytest.approx(62, abs=1.0)
    assert min(short) == pytest.approx(27, abs=1.0) and max(short) == pytest.approx(29, abs=1.0)
    assert math.sqrt(0.585 * 0.415 / 400) * 100 == pytest.approx(2.5, abs=0.1)


def test_the_example_path_of_the_defaults():
    rows = EV.analyse(S)
    path, n_two = EV.example_path(EV.example_row(rows, "Hin- und Rückspiel"), S)
    assert n_two == 16157 and len(path.seasons) == 3000 and path.first_seen == 116 and path.stays_out == 0


# --- Rauschgrenze --------------------------------------------------------------------------------------------------------------------------------------------

def test_noise_floor_of_eight_teams():
    """8 Teilnehmer, 0,3 logistische Stufe, ohne Remis, 4.000 Ligen je Wert (Zufalls-Seed 35): 42,3 / 52,3 / 63,1 / 78,4 / 88,0 / 95,3 % bei 1 / 2 / 4 / 10 / 20 / 40 Partien je Paar."""
    rows = EV.noise_rows(8, 35)
    assert [r["games"] for r in rows] == [1, 2, 4, 10, 20, 40]
    assert [round(100 * r["p_top"], 1) for r in rows] == pytest.approx([42.3, 52.3, 63.1, 78.4, 88.0, 95.3], abs=0.8)
    assert all(a["p_top"] < b["p_top"] for a, b in zip(rows, rows[1:]))
    assert 100 * rows[0]["se"] == pytest.approx(0.8, abs=0.05) and 100 * rows[-1]["se"] == pytest.approx(0.3, abs=0.05)      # README: Standardfehler 0,8 Pp bei 1 Partie, 0,3 Pp bei 40


def test_noise_floor_against_the_neighbour_demo():
    """Gegenprobe (README der Nachbar-Demo siegtabelle-demo, gelesen 2026-11-18; dort Direktvergleich bei Gleichstand, 200 feste Turniere, Stichprobenfehler 2 bis 3,5 Pp): etwa 40 / 65 / 75 / 93 % bei
    1 / 4 / 10 / 40 Partien je Paar; mit anderen 6.000 Seeds bei 1 Partie 43 %, bei 10 Partien 78 %. Hier mit 40.000 Ligen je Wert und Gleichstand zu gleichen Teilen."""
    big = {g: NO.top_chance(8, g, NO.ELO_STEP, 40000, 1 + i).p_top for i, g in enumerate((1, 4, 10, 40))}
    assert 100 * big[1] == pytest.approx(40, abs=4) and 100 * big[4] == pytest.approx(65, abs=4) and 100 * big[10] == pytest.approx(75, abs=4) and 100 * big[40] == pytest.approx(93, abs=4)
    assert [round(100 * big[g], 1) for g in (1, 4, 10, 40)] == pytest.approx([42.5, 63.2, 77.9, 95.2], abs=0.8)


def test_the_fair_level_of_the_plan_league_is_close_to_the_noise_floor_of_one_game():
    """Die Spielplan-Liga (8 Mannschaften, Elo-Abstand 300, Remisanteil 0,25, einfache Runde) hat ein Fairniveau von 42,0 % für die Stärkste, die Rauschgrenze mit einer Partie und 0,3 logistischer Stufe 42,3 %: beide nahe beieinander, aber Modelle mit eigenen Annahmen (Remis und Leiter gegen Partien ohne Remis)."""
    fair = EV.plan_row(8, 300, 20000, 101, 0, 0.5, False).p_good
    noise = NO.top_chance(8, 1, NO.ELO_STEP, 20000, 3).p_top
    assert 100 * fair == pytest.approx(42.0, abs=0.4) and 100 * noise == pytest.approx(42.4, abs=0.8)


# --- Presets -------------------------------------------------------------------------------------------------------------------------------------------------

def test_preset_strong_distortion_numbers():
    p = C.PRESETS["Starke Verzerrung"]
    s = EV.Settings(**p)
    table = {t["model"]: t for t in EV.table_rows(EV.analyse(s), s)}
    assert 100 * table["Hin- und Rückspiel"]["delta"] == pytest.approx(2.68, abs=0.005) and table["Hin- und Rückspiel"]["n"] == 4066
    assert 100 * table["Elfmeterschießen"]["delta"] == pytest.approx(21.11, abs=0.005) and table["Elfmeterschießen"]["n"] == 85
    assert 100 * table["Spielplan und Form"]["delta"] == pytest.approx(16.81, abs=0.8) and table["Spielplan und Form"]["n"] == pytest.approx(136, rel=0.15)
    assert 100 * table["Auftrittsreihenfolge"]["delta"] == pytest.approx(57.12, abs=1.0) and table["Auftrittsreihenfolge"]["n"] == pytest.approx(10, abs=1)


def test_preset_noise_only_has_no_difference_in_any_model():
    s = EV.Settings(**C.PRESETS["Nur Rauschen: keine Verzerrung"])
    rows = EV.analyse(s)
    assert [r.delta for r in rows] == [0.0, 0.0, 0.0, 0.0]
    assert all(math.isinf(EV.needed_for(r, 0.05, 0.8, "zwei Lagen")) for r in rows)


def test_preset_strict_test_numbers():
    s = EV.Settings(**C.PRESETS["Strenger Test: 1 % Fehlerniveau, 95 % Teststärke"])
    table = {t["model"]: t["n"] for t in EV.table_rows(EV.analyse(s), s)}
    assert table["Hin- und Rückspiel"] == 36669 and table["Elfmeterschießen"] == 1290
    assert table["Spielplan und Form"] == pytest.approx(4107, rel=0.12) and table["Auftrittsreihenfolge"] == pytest.approx(62, abs=5)


def test_preset_double_round_changes_only_the_plan_row():
    base = EV.analyse(S)
    s = EV.Settings(**C.PRESETS["Doppelrunde und mehr Partien je Paar"])
    rows = EV.analyse(s)
    assert rows[1:] == base[1:] or all(a.delta == b.delta for a, b in zip(rows[1:], base[1:]))
    assert 100 * rows[0].delta == pytest.approx(5.35, abs=0.8)


# --- Formel für N: Einzelwerte ------------------------------------------------------------------------------------------------------------------------------

def test_formula_numbers_for_the_two_group_design():
    """README: 60 % gegen 40 % (Δ = 20 Pp, Fairwert 50 %) braucht 97 Wettbewerbe je Lage (Fehlerniveau 5 %, Teststärke 80 %), 55 % gegen 45 % (Δ = 10 Pp) 392."""
    assert PW.whole_competitions(PW.n_two_groups(0.6, 0.4, 0.05, 0.8)) == 97
    assert PW.whole_competitions(PW.n_two_groups(0.55, 0.45, 0.05, 0.8)) == 392
