"""Wie viele beobachtete Wettbewerbe braucht man, um einen Lagenunterschied gegen den Zufall zu sehen?

Jeder beobachtete Wettbewerb liefert eine Ja-Nein-Größe: „die stärkste Seite gewinnt“ (Bernoulli, Varianz p·(1 - p): das ist das Rauschen). Zwei Messanordnungen:

* **zwei Lagen:** je N Wettbewerbe in der besten und N in der schlechtesten Lage; geprüft wird der Unterschied der Anteile (Zweistichproben-Test mit gemeinsamem Anteil
  unter der Nullhypothese, zweiseitig, Fehlerniveau α, Teststärke 1 - β);
* **gegen den Fairwert:** N Wettbewerbe in der besten Lage, geprüft wird der Anteil gegen den bekannten Fairwert p0 (Einstichproben-Test, zweiseitig). Das setzt voraus,
  dass man p0 kennt; in Wirklichkeit tut man das selten.

Die Zahl kommt aus der geschlossenen Form (Normalnäherung); `exact_power` rechnet die Teststärke derselben Tests exakt über beide Binomialverteilungen, `simulate_power`
würfelt den Test mit dem Zufallsgenerator der Demo. Nur Standardbibliothek und numpy.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from rau_scenario import uniform_block

DESIGNS = ("zwei Lagen", "gegen den Fairwert")
EXACT_N_LIMIT = 60000        # darüber rechnet die App die exakte Teststärke nicht mehr (Fenster wächst wie sqrt(N))


def z_quantile(prob: float) -> float:
    """Quantil der Standardnormalverteilung (Standardbibliothek)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob muss zwischen 0 und 1 liegen")
    return NormalDist().inv_cdf(prob)


def _check_levels(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha und power müssen zwischen 0 und 1 liegen")


def n_two_groups(p1: float, p2: float, alpha: float, power: float) -> float:
    """Wettbewerbe je Lage (zwei Lagen): N = (z_{1-α/2}·√(2·p̄·(1 - p̄)) + z_{power}·√(p1·(1 - p1) + p2·(1 - p2)))² / (p1 - p2)², p̄ = (p1 + p2)/2.

    Unendlich bei p1 = p2 (kein Unterschied ist nie zu sehen)."""
    _check_levels(alpha, power)
    if p1 == p2:
        return math.inf
    pbar = (p1 + p2) / 2.0
    num = z_quantile(1.0 - alpha / 2.0) * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_quantile(power) * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return num * num / (p1 - p2) ** 2


def n_against_fair(p0: float, p1: float, alpha: float, power: float) -> float:
    """Wettbewerbe in der besten Lage gegen den bekannten Fairwert p0: N = ((z_{1-α/2}·√(p0·(1 - p0)) + z_{power}·√(p1·(1 - p1))) / (p1 - p0))²."""
    _check_levels(alpha, power)
    if p1 == p0:
        return math.inf
    num = z_quantile(1.0 - alpha / 2.0) * math.sqrt(p0 * (1.0 - p0)) + z_quantile(power) * math.sqrt(p1 * (1.0 - p1))
    return (num / (p1 - p0)) ** 2


def n_needed(p_good: float, p_bad: float, alpha: float, power: float, design: str = "zwei Lagen") -> float:
    """Benötigte Zahl Wettbewerbe (ungerundet) je nach Messanordnung; der Fairwert ist bei `gegen den Fairwert` die Mitte (p_good + p_bad)/2, die Verschiebung also Δ/2."""
    if design == "zwei Lagen":
        return n_two_groups(p_good, p_bad, alpha, power)
    if design == "gegen den Fairwert":
        return n_against_fair((p_good + p_bad) / 2.0, p_good, alpha, power)
    raise ValueError(f"Messanordnung {design!r} unbekannt")


def whole_competitions(n_float: float) -> float:
    """Aufrunden auf ganze Wettbewerbe; unendlich bleibt unendlich."""
    return math.inf if math.isinf(n_float) else float(math.ceil(n_float - 1e-9))


# --- Binomialverteilung: Gewichte, exakte Teststärke, Ziehungen ----------------------------------------------------------------------------------------

def binomial_window(n: int, p: float, rel: float = 1e-17):
    """(erste Zahl, Gewichte) der Binomialverteilung Bin(n, p) auf dem Fenster, in dem das Gewicht mindestens `rel` mal das größte ist (Rest unter 1e-12).

    Die Gewichte werden als Summe von Logarithmen über das Verhältnis P(k)/P(k - 1) gerechnet und normiert."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    if p <= 0.0:
        return 0, np.array([1.0])
    if p >= 1.0:
        return n, np.array([1.0])
    k = np.arange(1, n + 1)
    logp = np.concatenate([[n * math.log1p(-p)], n * math.log1p(-p) + np.cumsum(np.log((n - k + 1.0) / k) + math.log(p / (1.0 - p)))])
    logp -= logp.max()
    keep = np.flatnonzero(logp > math.log(rel))
    lo, hi = int(keep[0]), int(keep[-1])
    w = np.exp(logp[lo : hi + 1])
    return lo, w / w.sum()


def rejects_two_groups(x1, x2, n: int, alpha: float):
    """Zweistichproben-Test: lehnt ab, wenn |p̂1 - p̂2| mindestens z_{1-α/2}·√(2·p̂·(1 - p̂)/n) ist (p̂ gemeinsamer Anteil); bei p̂ = 0 oder 1 keine Ablehnung."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    pooled = (x1 + x2) / (2.0 * n)
    se = np.sqrt(2.0 * pooled * (1.0 - pooled) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.abs(x1 - x2) / n / se
    return np.where(se > 0, stat >= z_quantile(1.0 - alpha / 2.0) - 1e-12, False)


def rejects_against_fair(x, n: int, p0: float, alpha: float):
    """Einstichproben-Test: lehnt ab, wenn |p̂ - p0| mindestens z_{1-α/2}·√(p0·(1 - p0)/n) ist."""
    x = np.asarray(x, dtype=float)
    se = math.sqrt(p0 * (1.0 - p0) / n)
    return np.abs(x / n - p0) >= z_quantile(1.0 - alpha / 2.0) * se - 1e-12


def exact_power(p_good: float, p_bad: float, n: int, alpha: float, design: str = "zwei Lagen") -> float:
    """Exakte Teststärke des Tests bei n Wettbewerben je Lage: Summe über alle Ausgänge beider Binomialverteilungen, bei denen der Test ablehnt."""
    n = int(n)
    if design == "zwei Lagen":
        lo1, w1 = binomial_window(n, p_good)
        lo2, w2 = binomial_window(n, p_bad)
        x1 = (lo1 + np.arange(len(w1)))[:, None]
        x2 = (lo2 + np.arange(len(w2)))[None, :]
        rej = rejects_two_groups(x1, x2, n, alpha)
        return float((w1[:, None] * w2[None, :] * rej).sum())
    if design == "gegen den Fairwert":
        lo, w = binomial_window(n, p_good)
        rej = rejects_against_fair(lo + np.arange(len(w)), n, (p_good + p_bad) / 2.0, alpha)
        return float((w * rej).sum())
    raise ValueError(f"Messanordnung {design!r} unbekannt")


def binomial_draws(n: int, p: float, u) -> np.ndarray:
    """Binomialzahlen Bin(n, p) per Inversion der Verteilungsfunktion aus gleichverteilten Zahlen `u` (exakt bis auf einen Rest unter 1e-12)."""
    lo, w = binomial_window(n, p)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return lo + np.minimum(np.searchsorted(cdf, np.asarray(u), side="right"), len(w) - 1)


@dataclass(frozen=True)
class PowerSim:
    rate: float      # Anteil der Wiederholungen, in denen der Test ablehnt
    se: float        # Standardfehler dieses Anteils
    reps: int


def simulate_power(p_good: float, p_bad: float, n: int, alpha: float, reps: int, seed: int, design: str = "zwei Lagen") -> PowerSim:
    """Würfelt `reps` Wiederholungen der Messung (je n Wettbewerbe je Lage) und zählt die Ablehnungen; Zufall: SplitMix64 mit `seed`."""
    if reps < 1:
        raise ValueError("reps muss mindestens 1 sein")
    u1 = uniform_block(seed, reps)
    x1 = binomial_draws(n, p_good, u1)
    if design == "zwei Lagen":
        x2 = binomial_draws(n, p_bad, uniform_block(seed, reps, start=reps))
        rej = rejects_two_groups(x1, x2, n, alpha)
    elif design == "gegen den Fairwert":
        rej = rejects_against_fair(x1, n, (p_good + p_bad) / 2.0, alpha)
    else:
        raise ValueError(f"Messanordnung {design!r} unbekannt")
    rate = float(np.mean(rej))
    return PowerSim(rate, math.sqrt(max(rate * (1.0 - rate), 1e-12) / reps), reps)


# --- Wachsendes Beispiel: Saison für Saison ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeasonPath:
    seasons: np.ndarray     # 1 ... S
    diff: np.ndarray        # Anteil der Siege der Stärksten in der guten Lage minus in der schlechten, nach s Saisons
    band: np.ndarray        # halbe Breite des Zufallsbandes bei s Saisons (z·√(2·p̂·(1 - p̂)/s), p̂ gemeinsamer Anteil)
    first_seen: int         # erste Saison s >= 2, in der |diff| das Band verlässt; 0, wenn nie
    stays_out: int          # erste Saison, ab der |diff| bis zur letzten Saison außerhalb des Bandes bleibt; 0, wenn die Linie am Ende im Band liegt


def season_path(p_good: float, p_bad: float, seasons: int, alpha: float, seed: int) -> SeasonPath:
    """Ein gewürfeltes Beispiel: In jeder Saison wird ein Wettbewerb in der guten und einer in der schlechten Lage gespielt (Sieg der Stärksten mit p_good bzw. p_bad)."""
    if seasons < 2:
        raise ValueError("Mindestens zwei Saisons")
    good = (uniform_block(seed, seasons) < p_good).astype(float)
    bad = (uniform_block(seed, seasons, start=seasons) < p_bad).astype(float)
    s = np.arange(1, seasons + 1)
    wg = np.cumsum(good)
    wb = np.cumsum(bad)
    diff = (wg - wb) / s
    pooled = (wg + wb) / (2.0 * s)
    band = z_quantile(1.0 - alpha / 2.0) * np.sqrt(2.0 * pooled * (1.0 - pooled) / s)
    out = (np.abs(diff) >= band) & (band > 0) & (s >= 2)
    first = int(s[np.argmax(out)]) if out.any() else 0
    inside = np.flatnonzero(~out)
    stays = 0 if (len(inside) and inside[-1] == seasons - 1) else (int(inside[-1]) + 2 if len(inside) else 1)
    return SeasonPath(s, diff, band, first, stays)


def peeking_rate(p: float, seasons: int, alpha: float, paths: int, seed: int) -> float:
    """Anteil gewürfelter Beispielwege **ohne jeden Lagenunterschied** (beide Lagen gewinnen mit p), die das Zufallsband mindestens einmal verlassen.

    So sieht es aus, wenn man die Linie jede Saison beobachtet und beim ersten Ausreißer abbricht: das Fehlerniveau α gilt für eine einzige Prüfung bei der vorab festgelegten Zahl,
    nicht für beliebig viele Prüfungen unterwegs. Jeder Weg hat seinen eigenen Strom (Zufalls-Seed + 1000·Nummer)."""
    if paths < 1:
        raise ValueError("paths muss mindestens 1 sein")
    hits = sum(season_path(p, p, seasons, alpha, seed + 1000 * k).first_seen > 0 for k in range(paths))
    return hits / paths
