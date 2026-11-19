"""Modell 2: Hin- und Rückspiel. Zwei Mannschaften spielen zweimal gegeneinander; gemessen wird, wie sehr das Heimrecht im Rückspiel die Chance der
**stärkeren** Mannschaft verschiebt, weiterzukommen.

Poisson-Tore, exakt gerechnet (Faltung zu zwei unabhängigen Differenzen), keine Simulation. Eigene, kompakte Fassung; sie importiert nichts aus anderen Demos.

Bezeichnungen: S ist die stärkere, W die schwächere Mannschaft (Elo-Abstand d >= 0 von S zu W). Hat S das Rückspiel zu Hause, ist S im Hinspiel Gast.
"""

import math
from dataclasses import dataclass

import numpy as np

ET_FRACTION = 1.0 / 3.0      # Verlängerung: 30 von 90 Minuten
TAIL = 1e-13                 # Rest der abgeschnittenen Poisson-Verteilung je Torzahl


def poisson_pmf(lam: float, tail: float = TAIL) -> np.ndarray:
    """Poisson-Gewichte P(k) für k = 0, 1, ... bis der Rest (Summe der weggelassenen) höchstens `tail` ist (Rekursion P(k+1) = P(k)·λ/(k+1))."""
    if lam < 0:
        raise ValueError("Die Torrate darf nicht negativ sein")
    if lam == 0:
        return np.array([1.0])
    p = [math.exp(-lam)]
    total = p[0]
    k = 0
    while 1.0 - total > tail and k < 400:
        k += 1
        p.append(p[-1] * lam / k)
        total += p[-1]
    return np.array(p)


def difference_pmf(lam_plus: float, lam_minus: float):
    """Verteilung von X - Y für unabhängige Poisson-Zahlen X ~ Poisson(lam_plus), Y ~ Poisson(lam_minus): (Werte, Gewichte), Faltung mit der gespiegelten Verteilung."""
    px = poisson_pmf(lam_plus)
    py = poisson_pmf(lam_minus)
    weights = np.convolve(px, py[::-1])
    values = np.arange(-(len(py) - 1), len(px))
    return values, weights


def team_factors(elo_gap: float):
    """Torratenfaktoren (stark, schwach) = (10^(d/800), 10^(-d/800)): das Torverhältnis ist 10^(d/400). Annahme der Demo, keine Messung."""
    return 10.0 ** (elo_gap / 800.0), 10.0 ** (-elo_gap / 800.0)


def extra_time_split(lam_home: float, lam_away: float):
    """Verlängerung im Stadion des Rückspiel-Gastgebers mit einem Drittel der Torrate: (P(Gastgeber mehr Tore), P(Gast mehr Tore), P(gleich viele, mindestens ein Tor je Seite), P(0:0))."""
    values, weights = difference_pmf(lam_home * ET_FRACTION, lam_away * ET_FRACTION)
    home = float(weights[values > 0].sum())
    away = float(weights[values < 0].sum())
    both_zero = math.exp(-(lam_home + lam_away) * ET_FRACTION)
    equal_scoring = float(weights[values == 0].sum()) - both_zero
    return home, away, equal_scoring, both_zero


@dataclass(frozen=True)
class DuelResult:
    p_strong: float          # Chance der stärkeren Mannschaft weiterzukommen
    p_decided_90: float      # Anteil der Duelle, die nach 2 x 90 Minuten (Summe, ggf. Auswärtstore) entschieden sind
    p_extra_time: float      # Anteil, der in die Verlängerung geht
    residual: float          # Masse, die das Abschneiden der Poisson-Verteilungen verliert (nur Anzeige und Test)


def advance_chance(lam_home: float, lam_away: float, elo_gap: float, strong_has_second_leg_home: bool, away_goals_rule: bool = False,
                   extra_time: bool = True, pen_strong: float = 0.5) -> DuelResult:
    """Chance der stärkeren Mannschaft S, weiterzukommen.

    Zerlegung: Die Auswärtstor-Differenz A = (Auswärtstore von S) - (Auswärtstore von W) und die Heim-Differenz H = (Heimtore von S) - (Heimtore von W) sind
    unabhängig; die Summe der Tore ist D = A + H. Entscheidung nach 2 x 90 Minuten: D > 0 S, D < 0 W; bei D = 0 mit Regel A > 0 S, A < 0 W; sonst
    Verlängerung (falls `extra_time`) im Stadion des Rückspiel-Gastgebers, sonst Elfmeterschießen, das S mit `pen_strong` gewinnt. In der Verlängerung
    entscheidet mehr Tore; bei Gleichstand mit mindestens einem Tor je Seite gewinnt mit Regel die Gastmannschaft, sonst und bei 0:0 das Elfmeterschießen."""
    if not 0.0 <= pen_strong <= 1.0:
        raise ValueError("pen_strong ist eine Wahrscheinlichkeit")
    f_s, f_w = team_factors(elo_gap)
    # Die Auswärtstor-Differenz A (S minus W) und die Heim-Differenz H hängen nicht davon ab, in welchem Spiel wer Gast ist: jede Mannschaft schießt
    # ihre Auswärtstore mit der Gast-Rate und ihre Heimtore mit der Heim-Rate, je einmal. Nur die Verlängerung hängt am Stadion.
    a_vals, a_w = difference_pmf(lam_away * f_s, lam_away * f_w)
    h_vals, h_w = difference_pmf(lam_home * f_s, lam_home * f_w)
    residual = 1.0 - float(a_w.sum()) * float(h_w.sum())
    grid_a, grid_h = np.meshgrid(a_vals, h_vals, indexing="ij")
    weight = np.outer(a_w, h_w)
    total = grid_a + grid_h
    s_wins = float(weight[total > 0].sum())
    level = total == 0
    if away_goals_rule:
        s_wins += float(weight[level & (grid_a > 0)].sum())
        open_mass = float(weight[level & (grid_a == 0)].sum())
    else:
        open_mass = float(weight[level].sum())
    p = s_wins
    if open_mass > 0.0:
        if extra_time:
            host_f, guest_f = (f_s, f_w) if strong_has_second_leg_home else (f_w, f_s)
            host_win, guest_win, equal_scoring, both_zero = extra_time_split(lam_home * host_f, lam_away * guest_f)
            s_et, pens = (host_win, both_zero) if strong_has_second_leg_home else (guest_win, both_zero)
            if away_goals_rule:
                s_et += 0.0 if strong_has_second_leg_home else equal_scoring   # der Gast gewinnt bei gleich vielen Toren (mindestens eins je Seite)
            else:
                pens += equal_scoring
            p += open_mass * (s_et + pens * pen_strong)
        else:
            p += open_mass * pen_strong
    p_extra = open_mass if extra_time else 0.0
    return DuelResult(p, 1.0 - open_mass, p_extra, residual)


def lage_effect(lam_home: float, lam_away: float, elo_gap: float, away_goals_rule: bool = False, extra_time: bool = True, pen_strong: float = 0.5):
    """(Chance der stärkeren Mannschaft mit dem Rückspiel zu Hause, ohne es) und ihre Differenz (der Wert des Rückspiel-Heimrechts für die stärkere Seite)."""
    good = advance_chance(lam_home, lam_away, elo_gap, True, away_goals_rule, extra_time, pen_strong).p_strong
    bad = advance_chance(lam_home, lam_away, elo_gap, False, away_goals_rule, extra_time, pen_strong).p_strong
    return good, bad, good - bad
