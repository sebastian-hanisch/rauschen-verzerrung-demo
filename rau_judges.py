"""Modell 3: Auftrittsreihenfolge bei Kampfrichter-Noten. Auftretende treten nacheinander auf, Kampfrichter geben Noten; die Noten enthalten neben der Qualität
eine Verzerrung durch die Position (Drift je Platz) und durch die vorherige Panelnote (Angleichung oder Kontrast). Gemessen wird, wie sehr die Auftrittsnummer die
Chance verschiebt, dass die **Qualitätsbeste** gewinnt.

Simulation mit gemeinsamen Zufallszahlen über alle Auftrittsnummern: Qualität, Rauschen und Zufallsreihenfolge der übrigen hängen am Auftretenden, nicht am Platz.
Eigene, kompakte Fassung; sie importiert nichts aus anderen Demos.

Note des Auftretenden auf Platz k (von 0 gezählt) von Richter j:  x[k, j] = q + σ·e[k, j] + β·(k - (n - 1)/2) + γ·A[k - 1],  A[k] = Mittel der Noten von Platz k,  A[-1] = 0.
Wertung = Mittel der Noten; es gewinnt die höchste Wertung.
"""

import numpy as np

from rau_lage import Lage, split_best_worst
from rau_scenario import normal_block, uniform_block


def draw_competitions(n: int, judges: int, runs: int, seed: int):
    """Zufallszahlen für `runs` Wettbewerbe: Qualität q (runs, n), Rauschen e (runs, n, judges) und Mischschlüssel (runs, n) für die Reihenfolge der Übrigen."""
    q = normal_block(seed, runs * n).reshape(runs, n)
    e = normal_block(seed, runs * n * judges, start=runs * n).reshape(runs, n, judges)
    keys = uniform_block(seed, runs * n, start=2 * (runs * n + runs * n * judges)).reshape(runs, n)
    return q, e, keys


def order_with_strongest_at(q: np.ndarray, keys: np.ndarray, position: int) -> np.ndarray:
    """Auftrittsreihenfolge (runs, n): auf Platz `position` steht die Qualitätsbeste, die übrigen stehen in der Reihenfolge ihrer Schlüssel auf den anderen Plätzen."""
    runs, n = q.shape
    if not 0 <= position < n:
        raise ValueError("position muss ein Platz von 0 bis n - 1 sein")
    best = q.argmax(axis=1)
    masked = keys.copy()
    masked[np.arange(runs), best] = np.inf              # die Beste kommt in der Sortierung zuletzt und wird abgeschnitten
    others = np.argsort(masked, axis=1, kind="stable")[:, : n - 1]
    order = np.empty((runs, n), dtype=np.int64)
    order[:, :position] = others[:, :position]
    order[:, position] = best
    order[:, position + 1 :] = others[:, position:]
    return order


def panel_scores(q_pos: np.ndarray, e_pos: np.ndarray, sigma: float, drift: float, anchor: float) -> np.ndarray:
    """Wertung (Mittel der Noten) je Platz, Form (runs, n); `q_pos` und `e_pos` sind schon in Auftrittsreihenfolge (Platz 0 zuerst)."""
    runs, n = q_pos.shape
    ratings = np.empty((runs, n))
    previous = np.zeros(runs)
    for k in range(n):
        notes = q_pos[:, k, None] + sigma * e_pos[:, k, :] + drift * (k - (n - 1) / 2.0) + anchor * previous[:, None]
        ratings[:, k] = notes.mean(axis=1)
        previous = ratings[:, k]
    return ratings


def strongest_wins(n: int, judges: int, sigma: float, drift: float, anchor: float, runs: int, seed: int, positions=None) -> np.ndarray:
    """Array (Plätze, Läufe): 1, wenn die Qualitätsbeste auf diesem Platz gewinnt, sonst 0 (gemeinsame Zufallszahlen über die Plätze).

    `positions` wählt die Plätze (Standard: alle n)."""
    if n < 2:
        raise ValueError("Ein Wettbewerb braucht mindestens zwei Auftretende")
    if judges < 1:
        raise ValueError("Mindestens ein Kampfrichter")
    q, e, keys = draw_competitions(n, judges, runs, seed)
    places = range(n) if positions is None else positions
    out = []
    for position in places:
        order = order_with_strongest_at(q, keys, position)
        q_pos = np.take_along_axis(q, order, axis=1)
        e_pos = np.take_along_axis(e, order[:, :, None], axis=1)
        ratings = panel_scores(q_pos, e_pos, sigma, drift, anchor)
        out.append((ratings.argmax(axis=1) == position).astype(np.float64))
    return np.array(out)


def auftritt_lage(n: int, judges: int, sigma: float, drift: float, anchor: float, runs: int, seed: int) -> tuple[Lage, np.ndarray]:
    """Beste und schlechteste Auftrittsnummer für die Qualitätsbeste (ausgewählt an der ersten Hälfte der Läufe, gemessen an der zweiten) und die Chance je Platz (alle Läufe)."""
    wins = strongest_wins(n, judges, sigma, drift, anchor, runs, seed)
    return split_best_worst(wins), wins.mean(axis=1)
