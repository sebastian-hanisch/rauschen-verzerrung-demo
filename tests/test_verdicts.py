"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier."""

import math

import numpy as np

import rau_evaluation as ev
import rau_power as PW


def _row(model, n, delta, exact=True, n_low=None, n_high=None, **extra):
    base = {"model": model, "good": "a", "bad": "b", "p_good": 0.6, "p_bad": 0.5, "p_fair": 0.55, "noise": 0.45, "delta": delta, "se": float("nan") if exact else 0.01,
            "exact": exact, "n": n, "n_low": n if n_low is None else n_low, "n_high": n if n_high is None else n_high}
    base.update(extra)
    return base


# --- Anzeigehilfen ---------------------------------------------------------------------------------------------------------------------------------------

def test_german_number_formats():
    assert ev.de(1.5, 2) == "1,50"
    assert ev.pct(0.4219) == "42,2 %" and ev.pct(0.5, 0) == "50 %"
    assert ev.pp(0.0135) == "+1,35 Pp" and ev.pp(-0.0058) == "−0,58 Pp" and ev.pp(0.0) == "0,00 Pp" and ev.pp(-1e-9) == "0,00 Pp"
    assert ev.count(16157.0) == "16.157" and ev.count(28.0) == "28" and ev.count(1234567.0) == "1.234.567"
    assert ev.count(math.inf) == "unbestimmt (Δ = 0)"


# --- Vergleich der vier Modelle ----------------------------------------------------------------------------------------------------------------------------

def test_comparison_names_the_easiest_and_the_hardest_with_the_unit():
    table = [_row("Spielplan und Form", 1810.0, 0.0461, False), _row("Hin- und Rückspiel", 16157.0, 0.0135), _row("Auftrittsreihenfolge", 28.0, 0.3137, False), _row("Elfmeterschießen", 569.0, 0.0819)]
    kind, text = ev.verdict_comparison(table)
    assert kind == "success"                                                      # Verhältnis 577 >= 10
    assert "Einheit: Δ ist der Unterschied der Chance der stärksten Seite" in text and "Prozentpunkten (Pp)" in text and "N zählt beobachtete Wettbewerbe" in text
    assert "Am leichtesten fällt **Auftrittsreihenfolge** auf (Δ = +31,37 Pp, N = 28)" in text
    assert "am schwersten **Hin- und Rückspiel** (Δ = +1,35 Pp, N = 16.157)" in text
    assert "Das Verhältnis der beiden Zahlen ist 577,0." in text


def test_comparison_names_the_single_sample_of_the_plan_model_with_the_measured_seed_spread():
    table = [_row("Spielplan und Form", 1810.0, 0.0461, False), _row("Hin- und Rückspiel", 16157.0, 0.0135)]
    text = ev.verdict_comparison(table)[1]
    assert "Beim Spielplan ist Δ eine Einzelstichprobe" in text and "0,98 Pp gegen 0,54 Pp" in text and "Zufalls-Seeds 101 bis 105" in text
    assert text.endswith("Das Verhältnis der beiden Zahlen ist 8,9.")
    without = ev.verdict_comparison([_row("Auftrittsreihenfolge", 28.0, 0.3137, False), _row("Hin- und Rückspiel", 16157.0, 0.0135)])[1]
    assert "Einzelstichprobe" not in without                                      # nur wenn der Spielplan in der Tabelle steht
    flat = ev.verdict_comparison([_row("Spielplan und Form", math.inf, 0.0, False), _row("Hin- und Rückspiel", 16157.0, 0.0135)])[1]
    assert "Einzelstichprobe" not in flat                                         # ohne Lagenunterschied im Spielplan nichts zu messen


def test_comparison_threshold_is_ten():
    table = [_row("A", 100.0, 0.2), _row("B", 1000.0, 0.1)]
    assert ev.verdict_comparison(table)[0] == "success"                           # genau 10
    table = [_row("A", 100.0, 0.2), _row("B", 950.0, 0.1)]
    kind, text = ev.verdict_comparison(table)
    assert kind == "info" and "ist 9,5." in text


def test_comparison_lists_models_without_difference():
    table = [_row("A", 100.0, 0.2), _row("B", math.inf, 0.0)]
    kind, text = ev.verdict_comparison(table)
    assert "Ohne Unterschied (N unbestimmt): B." in text


def test_comparison_without_any_difference_is_the_noise_message():
    table = [_row("A", math.inf, 0.0), _row("B", math.inf, 0.0)]
    kind, text = ev.verdict_comparison(table)
    assert kind == "info" and "In allen vier Modellen" in text and "unbestimmt" in text


def test_comparison_names_models_whose_standard_error_allows_zero():
    table = [_row("A", 100.0, 0.2), _row("S", 500.0, 0.03, False, 300.0, math.inf)]
    kind, text = ev.verdict_comparison(table)
    assert "Der Standardfehler lässt bei S zu, dass Δ null ist" in text and "nach oben offen" in text


# --- Wachsendes Beispiel -----------------------------------------------------------------------------------------------------------------------------------

def _path(first, stays, seasons=300):
    s = np.arange(1, seasons + 1)
    return PW.SeasonPath(s, np.zeros(seasons), np.ones(seasons), first, stays)


def test_example_verdict_branches():
    kind, text = ev.verdict_example("Hin- und Rückspiel", _path(116, 0), 16157.0, 3000, 0.05)
    assert kind == "success" and "N = 16.157 Wettbewerbe je Lage (im Beispiel je „Saison“ ein Wettbewerb je Lage)" in text and "(3000 Saisons gezeigt)" in text
    assert "erstmals in Saison 116" in text and "liegt aber am Ende wieder im Band" in text
    kind, text = ev.verdict_example("Elfmeterschießen", _path(40, 55), 569.0, 1707, 0.05)
    assert kind == "success" and "erstmals in Saison 40 und bleibt ab Saison 55 draußen" in text
    kind, text = ev.verdict_example("Elfmeterschießen", _path(0, 0), 569.0, 1707, 0.05)
    assert kind == "warning" and "verlässt die Linie das Zufallsband nicht" in text
    kind, text = ev.verdict_example("Spielplan und Form", _path(0, 0), math.inf, 3000, 0.05)
    assert kind == "info" and "keinen Lagenunterschied" in text and "Fehlerniveau 5 %" in text


# --- Rauschgrenze ------------------------------------------------------------------------------------------------------------------------------------------

def test_noise_verdict_names_first_and_last_value():
    rows = [{"games": 1, "p_top": 0.423, "se": 0.008}, {"games": 4, "p_top": 0.63, "se": 0.008}, {"games": 40, "p_top": 0.953, "se": 0.003}]
    kind, text = ev.verdict_noise(rows)
    assert kind == "info"
    assert "bei 1 Partie(n) je Paar nur in 42,3 % der Ligen" in text and "Standardfehler 0,8 Pp" in text and "bei 40 Partien je Paar in 95,3 %" in text


# --- Probe der Teststärke ------------------------------------------------------------------------------------------------------------------------------------

def test_power_verdict_threshold_is_five_points():
    close = PW.PowerSim(0.84, 0.006, 4000)
    kind, text = ev.verdict_power(close, 0.8, 0.8007)
    assert kind == "success" and "84,0 %" in text and "von 4.000" in text and "Ziel 80 %" in text and "Exakt über beide Binomialverteilungen: 80,1 %" in text
    kind, text = ev.verdict_power(PW.PowerSim(0.85, 0.006, 4000), 0.8, None)
    assert kind == "success" and "Exakt über" not in text                          # genau 5 Pp: noch in Ordnung
    kind, text = ev.verdict_power(PW.PowerSim(0.86, 0.006, 4000), 0.8, None)
    assert kind == "warning" and "um mehr als 5 Pp" in text


# --- Messreihe ----------------------------------------------------------------------------------------------------------------------------------------------

def test_sweep_verdict_names_smallest_and_largest_n_and_skips_zero_delta():
    series = {"A": [(0.0, math.inf), (0.2, 90.0)], "B": [(0.01, 20000.0), (0.0, math.inf)]}
    kind, text = ev.verdict_sweep(series)
    assert kind == "info"
    assert "von 90 (A, Δ = +20,00 Pp) bis 20.000 (B, Δ = +1,00 Pp)" in text and "1/Δ²" in text
    kind, text = ev.verdict_sweep({"A": [(0.0, math.inf)]})
    assert "Keine Reihe" in text
