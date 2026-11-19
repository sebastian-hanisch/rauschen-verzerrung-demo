"""Gemeinsame Einheit der Demo: die **Lage** einer Seite und der Unterschied ihrer Gewinnchance zwischen bester und schlechtester Lage.

Eine Lage ist eine Ausgangsposition, die nichts mit der Stärke zu tun hat: das Heimrecht im Rückspiel, der Platz in der Zirkeltafel, die Auftrittsnummer, die
Schussreihenfolge. Gemessen wird immer dieselbe Größe: die Chance der **wahrhaft stärksten** Seite, zu gewinnen, wenn sie die beste Lage hat, minus dieselbe
Chance bei der schlechtesten Lage (Prozentpunkte). Sie heißt hier Lagenunterschied Δ.
"""

from dataclasses import dataclass

import numpy as np

DELTA_ZERO = 1e-9    # kleinere Unterschiede exakt gerechneter Chancen sind Rundungsreste und gelten als null


@dataclass(frozen=True)
class Lage:
    """Beste und schlechteste Lage mit ihren gemessenen Gewinnchancen; Anteile, nicht Prozent."""

    best: int             # Index der besten Lage
    worst: int            # Index der schlechtesten Lage
    p_best: float
    p_worst: float
    delta: float          # p_best - p_worst
    se: float             # gepaarter Standardfehler von delta (nan bei exakten Werten)
    n_measure: int        # Zahl der Läufe der Messung (0 bei exakten Werten)


def exact_lage(chances) -> Lage:
    """Lagenunterschied aus exakt berechneten Chancen je Lage (keine Simulation): größte minus kleinste Chance."""
    chances = np.asarray(chances, dtype=float)
    best = int(np.argmax(chances))
    worst = int(np.argmin(chances))
    delta = float(chances[best] - chances[worst])
    if delta < DELTA_ZERO:
        delta = 0.0
    return Lage(best, worst, float(chances[best]), float(chances[worst]), delta, float("nan"), 0)


def split_best_worst(outcome, select=None) -> Lage:
    """Beste und schlechteste Lage aus Simulationsläufen, ohne dass die Auswahl die Messung verfälscht.

    `outcome[k, r]` ist das Ergebnis der stärksten Seite in Lage k und Lauf r (1 = gewinnt; bei Punktgleichheit ein Anteil), gemeinsame Zufallszahlen über k.
    Ausgewählt wird an der ersten Hälfte der Läufe (nach `select`, ersatzweise nach `outcome`; höher = besser), gemessen an der zweiten: eine Auswahl, die
    Rauschen als Vorteil liest, kann die gemessene Differenz dann nicht nach oben treiben (sie schrumpft eher). Der Standardfehler ist gepaart (gleiche Läufe)."""
    outcome = np.asarray(outcome, dtype=float)
    if outcome.ndim != 2 or outcome.shape[1] < 2:
        raise ValueError("outcome braucht die Form (Lagen, Läufe) mit mindestens zwei Läufen")
    runs = outcome.shape[1]
    half = runs // 2
    score = outcome if select is None else np.asarray(select, dtype=float)
    mean_select = score[:, :half].mean(axis=1)
    best = int(np.argmax(mean_select))
    worst = int(np.argmin(mean_select))
    a = outcome[best, half:]
    b = outcome[worst, half:]
    d = a - b
    n_b = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n_b)) if n_b > 1 else float("nan")
    return Lage(best, worst, float(a.mean()), float(b.mean()), float(d.mean()), se, n_b)
