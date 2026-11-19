"""Szenario-Bausteine, die mehrere Modelle der Demo teilen: Zufallsblöcke, Elo-Umrechnung, Zirkeltafel, Zuordnungen.

Alle Zufallszahlen stammen aus SplitMix64 (`rau_rng.py`, plattformfest); `uniform_block` rechnet dieselbe Folge wie `SplitMix64.random()` vektorisiert
(Test: bitgleich). Normalzahlen entstehen per Box-Muller mit numpy-Funktionen; sie können sich von der `decimal`-Fassung der Klasse in der letzten Binärstelle
unterscheiden, die Tests prüfen sie mit Band.
"""

import math

import numpy as np

from rau_rng import SplitMix64

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MASK64 = (1 << 64) - 1


def uniform_block(seed: int, count: int, start: int = 0) -> np.ndarray:
    """Die Werte Nummer `start` bis `start + count - 1` von SplitMix64(seed).random() als Array (vektorisiert, bitgleich zur Klasse).

    Zustand k = seed + (k + 1)·γ (mod 2^64), danach die Mischfunktion; `start` springt über die ersten Werte."""
    k = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & _MASK64) + k * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def normal_block(seed: int, count: int, start: int = 0) -> np.ndarray:
    """Standardnormalzahlen aus je zwei aufeinanderfolgenden gleichverteilten Zahlen (Box-Muller, nur der Kosinus-Zweig); `count` Zahlen brauchen 2·count Werte."""
    u = uniform_block(seed, 2 * count, 2 * start)
    u1 = np.maximum(u[0::2], 1e-12)
    u2 = u[1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def elo_expectation(diff):
    """Elo-Erwartung 1 / (1 + 10^(-Δ/400)) für einen Stärkeunterschied Δ in Elo-Punkten (Zahl oder Array)."""
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(diff, dtype=float) / 400.0))


def elo_ladder(n: int, gap: float) -> np.ndarray:
    """n Stärken gleichmäßig von +gap/2 (Mannschaft 0, die stärkste) bis -gap/2; bei n = 1 oder gap = 0 alle 0."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    if n == 1:
        return np.zeros(1)
    return gap * (0.5 - np.arange(n) / (n - 1))


def circle_rounds(n: int) -> list[list[tuple[int, int]]]:
    """Zirkelmethode (Berger-Tafel) für gerades n: n - 1 Spieltage mit je n/2 Paaren aus Plätzen 0 bis n - 1; Platz 0 bleibt, die übrigen rücken weiter."""
    if n < 2 or n % 2:
        raise ValueError("Die Zirkelmethode braucht eine gerade Mannschaftszahl von mindestens 2")
    rest = list(range(1, n))
    rounds = []
    for t in range(n - 1):
        turn = rest[-t:] + rest[:-t] if t else rest[:]
        order = [0] + turn
        rounds.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
    return rounds


def random_assignments(n: int, count: int, seed: int) -> np.ndarray:
    """`count` Zuordnungen der Mannschaften zu den Plätzen der Zirkeltafel als Array (count, n): Zeile k setzt auf Platz p die Mannschaft `perm[k, p]`.

    Die erste Zeile ist die Identität (Mannschaft p auf Platz p), die übrigen sind Mischungen (Fisher-Yates mit SplitMix64); bei n! <= count nur die verschiedenen, alle n!."""
    if count < 1:
        raise ValueError("count muss mindestens 1 sein")
    rng = SplitMix64(seed)
    rows = [tuple(range(n))]
    seen = {rows[0]}
    total = math.factorial(n)
    target = min(count, total)
    guard = 0
    while len(rows) < target and guard < 200 * count:
        guard += 1
        items = list(range(n))
        rng.shuffle(items)
        key = tuple(items)
        if key not in seen:
            seen.add(key)
            rows.append(key)
    return np.array(rows, dtype=np.int64)
