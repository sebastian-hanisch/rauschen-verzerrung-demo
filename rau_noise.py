"""Die Rauschgrenze: Wie oft steht die wahrhaft stärkste Seite am Ende einer Liga ohne jede Verzerrung auf Platz 1?

Das ist der Vergleichsmaßstab der Demo: Auch ein vollständig fairer Wettbewerb findet die Stärkste nicht immer; der Rest ist Zufall. Die Rechnung ist eine eigene,
kleine Fassung (Liga mit k Partien je Paar, Stärken auf einer Elo-Leiter, Tabelle nach Punkten, Gleichstand an der Spitze zu gleichen Teilen); sie importiert nichts
aus anderen Demos. Standard sind Partien ohne Remis und ein Stärkeabstand von 0,3 in der logistischen Stärke je Platz (das sind rund 52 Elo-Punkte).
"""

import math
from dataclasses import dataclass

import numpy as np

from rau_plan import draw_split
from rau_scenario import elo_expectation, uniform_block

LOGISTIC_STEP = 0.3                                    # Stärkeabstand je Platz in der logistischen Stärke (Bradley-Terry)
ELO_STEP = LOGISTIC_STEP * 400.0 / math.log(10.0)       # dasselbe in Elo-Punkten: e = 1/(1 + exp(-Δ_logistisch)) = 1/(1 + 10^(-Δ_elo/400))


@dataclass(frozen=True)
class NoiseResult:
    p_top: float        # Anteil der Ligen, in denen die Stärkste an der Spitze steht (Gleichstand zu gleichen Teilen)
    se: float           # Standardfehler dieses Anteils (Läufe als unabhängig gerechnet)
    runs: int


def pair_indices(n: int):
    """Alle Paare i < j einer Liga als zwei Indexfelder."""
    i, j = np.triu_indices(n, k=1)
    return i, j


def top_chance(n: int, games: int, elo_step: float, runs: int, seed: int, draw_max: float = 0.0) -> NoiseResult:
    """Chance, dass die stärkste von n Mannschaften (Elo-Abstand `elo_step` je Platz) eine Liga mit `games` Partien je Paar an der Spitze beendet.

    Punkte 3 / 1 / 0; Gleichstand an der Spitze: die Stärkste zählt 1/k bei k Punktgleichen (der Erwartungswert eines Loses)."""
    if n < 2 or games < 1 or runs < 1:
        raise ValueError("n >= 2, games >= 1 und runs >= 1 sind nötig")
    i, j = pair_indices(n)
    p = len(i)
    ratings = -elo_step * np.arange(n)
    win, draw = draw_split(elo_expectation(ratings[i] - ratings[j]), draw_max)
    u = uniform_block(seed, runs * p * games).reshape(runs, p, games)
    wins_i = (u < win[None, :, None]).sum(axis=2)
    draws = ((u >= win[None, :, None]) & (u < (win + draw)[None, :, None])).sum(axis=2)
    pts_i = 3 * wins_i + draws
    pts_j = 3 * (games - wins_i - draws) + draws
    one_i = np.zeros((p, n))
    one_j = np.zeros((p, n))
    one_i[np.arange(p), i] = 1.0
    one_j[np.arange(p), j] = 1.0
    pts = pts_i @ one_i + pts_j @ one_j
    top = pts.max(axis=1)
    at_top = pts == top[:, None]
    share = at_top[:, 0] / at_top.sum(axis=1)
    mean = float(share.mean())
    return NoiseResult(mean, math.sqrt(max(mean * (1.0 - mean), 1e-12) / runs), runs)


def noise_curve(n: int, games_list, elo_step: float, runs: int, seed: int, draw_max: float = 0.0) -> list[NoiseResult]:
    """`top_chance` für mehrere Zahlen von Partien je Paar (gleicher Zufalls-Seed, aber je Zahl ein eigener Strom)."""
    return [top_chance(n, g, elo_step, runs, seed + 17 * idx, draw_max) for idx, g in enumerate(games_list)]
