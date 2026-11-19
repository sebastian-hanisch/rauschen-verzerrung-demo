"""Modell 1: Spielplan und Form. Eine Liga nach der Zirkelmethode, in der jede Mannschaft einen Zustand („Form“) hat; gemessen wird, wie sehr die
Zuordnung der Mannschaften zu den Plätzen der Zirkeltafel die Titelchance der stärksten Mannschaft verschiebt.

Alle Zahlen sind Simulationen mit gemeinsamen Zufallszahlen: Jedes Paar hat in jedem Lauf (und je Hin-/Rückspiel) eine eigene Zufallszahl, jede Mannschaft
je Spieltag eine Münze für den Zustand und eine für den Start; sie hängen an Paar und Mannschaft, nicht am Platz der Tafel und gehen in jede Zuordnung gleich ein.
Ohne Wirkung der Form (m = 0) ist darum jede Zuordnung dasselbe Ergebnis, Lauf für Lauf.

Dieses Modell ist eine eigene, kompakte Fassung; es importiert nichts aus anderen Demos.
"""

from dataclasses import dataclass

import numpy as np

from rau_lage import Lage, split_best_worst
from rau_scenario import circle_rounds, elo_expectation, elo_ladder, random_assignments, uniform_block

STATES = np.array([-1, 0, 1])   # kalt, normal, heiß


def draw_split(expectation, draw_max: float):
    """Zerlegt die Elo-Erwartung e in Sieg- und Remisanteil: Remis d = min(draw_max, 2·min(e, 1 - e)), Sieg e - d/2 (die erwartete Punktquote bleibt e)."""
    e = np.asarray(expectation, dtype=float)
    d = np.minimum(draw_max, 2.0 * np.minimum(e, 1.0 - e))
    return e - d / 2.0, d


def result_probabilities(rating_a, rating_b, draw_max: float):
    """(Sieg, Remis, Niederlage) von A gegen B, die Stärken als Elo-Zahlen (Arrays dürfen sich verbreiten)."""
    win, draw = draw_split(elo_expectation(np.asarray(rating_a) - np.asarray(rating_b)), draw_max)
    return win, draw, 1.0 - win - draw


def form_start_distribution(ratings, m: float, rho: float, draw_max: float, max_iter: int = 2000, tol: float = 1e-13):
    """Startverteilung des Zustands je Mannschaft (n × 3, Spalten kalt/normal/heiß): Fixpunkt im Mittelfeld-Modell.

    Die Kette einer Mannschaft i ist P = ρ·I + (1 - ρ)·Q mit Q[s, s'] = Wahrscheinlichkeit, dass das Ergebnis den Zustand s' setzt, wenn i im Zustand s gegen
    eine gleichverteilt gewählte andere Mannschaft j in deren Verteilung π_j spielt. π = πP ist der Fixpunkt über alle Mannschaften gemeinsam
    (πP = π gilt genau dann, wenn πQ = π, die Verteilung hängt also nicht an ρ). Näherung: die Gegner einer Saison sind keine unabhängigen Ziehungen."""
    r = np.asarray(ratings, dtype=float)
    n = len(r)
    pi = np.full((n, 3), 1.0 / 3.0)
    if n < 2:
        return pi
    # D[i, j, s, t]: Stärkeunterschied i gegen j, wenn i im Zustand s und j im Zustand t ist
    diff = (r[:, None, None, None] - r[None, :, None, None]) + m * (STATES[None, None, :, None] - STATES[None, None, None, :])
    win, draw, loss = result_probabilities(diff, 0.0, draw_max)
    others = 1.0 - np.eye(n)
    for _ in range(max_iter):
        w = others[:, :, None] * pi[None, :, :] / (n - 1)           # Gewicht von (j, t) für i
        q = np.stack([np.einsum("ijt,ijst->is", w, loss), np.einsum("ijt,ijst->is", w, draw), np.einsum("ijt,ijst->is", w, win)], axis=2)  # (n, s, Ergebnis)
        p = rho * np.eye(3)[None] + (1.0 - rho) * q
        new = np.einsum("is,isk->ik", pi, p)
        new /= new.sum(axis=1, keepdims=True)                       # Rundungsreste der Masse würden sich sonst von Schritt zu Schritt vervielfachen
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    return pi


@dataclass(frozen=True)
class PlanRun:
    """Ergebnis einer Simulation über mehrere Zuordnungen: für jede Zuordnung k und jeden Lauf r die Titelchance der stärksten Mannschaft
    (`share`: 1 bei Alleinsieg, 1/k bei k Punktgleichen an der Spitze) und ihre Punkte (`points`)."""

    share: np.ndarray    # (Zuordnungen, Läufe), float32
    points: np.ndarray   # (Zuordnungen, Läufe), int16


def simulate_plans(ratings, perms, runs: int, seed: int, m: float, rho: float, draw_max: float, double: bool = False, start=None) -> PlanRun:
    """Spielt `runs` Ligen je Zuordnung (Zeilen von `perms`) mit gemeinsamen Zufallszahlen und gibt die Titelchance und die Punkte der Mannschaft 0 zurück.

    Zustand s ∈ {-1, 0, +1}: Stärke = Grundstärke + m·s; nach jeder Partie bleibt er mit Wahrscheinlichkeit ρ, sonst wird er auf das Ergebnis gesetzt
    (Sieg +1, Remis 0, Niederlage -1). Start aus `start` (Standard: `form_start_distribution`). Punkte 3 / 1 / 0."""
    r = np.asarray(ratings, dtype=float)
    perms = np.asarray(perms)
    n = len(r)
    if runs < 1:
        raise ValueError("runs muss mindestens 1 sein")
    rounds = circle_rounds(n)
    legs = 2 if double else 1
    slots = np.array(rounds * legs)                          # (T, g, 2)
    t_total, g, _ = slots.shape
    per_run = legs * n * n + t_total * n + n
    block = uniform_block(seed, runs * per_run).reshape(runs, per_run)
    u_pair = block[:, : legs * n * n].reshape(runs, legs, n * n)
    u_coin = block[:, legs * n * n : legs * n * n + t_total * n].reshape(runs, t_total, n)
    u_start = block[:, -n:]
    pi = form_start_distribution(r, m, rho, draw_max) if start is None else np.asarray(start, dtype=float)
    cdf = np.cumsum(pi, axis=1)
    s0 = np.where(u_start < cdf[None, :, 0], -1, np.where(u_start < cdf[None, :, 1], 0, 1)).astype(np.int8)

    share = np.zeros((len(perms), runs), dtype=np.float32)
    points = np.zeros((len(perms), runs), dtype=np.int16)
    for k, perm in enumerate(perms):
        state = s0.copy()
        pts = np.zeros((runs, n), dtype=np.int16)
        for t in range(t_total):
            a = perm[slots[t, :, 0]]
            b = perm[slots[t, :, 1]]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            u = u_pair[:, t // (n - 1), lo * n + hi]          # (Läufe, g)
            diff = (r[lo] - r[hi])[None, :] + m * (state[:, lo].astype(float) - state[:, hi].astype(float))
            win, draw = draw_split(elo_expectation(diff), draw_max)
            res_lo = np.where(u < win, 1, np.where(u < win + draw, 0, -1))
            pts[:, lo] += (1 + res_lo + (res_lo == 1)).astype(np.int16)
            pts[:, hi] += (1 - res_lo + (res_lo == -1)).astype(np.int16)
            res = np.empty((runs, n), dtype=np.int8)
            res[:, lo] = res_lo
            res[:, hi] = -res_lo
            state = np.where(u_coin[:, t, :] < rho, state, res)
        top = pts.max(axis=1)
        at_top = pts == top[:, None]
        share[k] = (at_top[:, 0] / at_top.sum(axis=1)).astype(np.float32)
        points[k] = pts[:, 0]
    return PlanRun(share, points)


def points_spread(run: PlanRun) -> float:
    """Größter minus kleinster Mittelwert der Punkte der stärksten Mannschaft über die Zuordnungen (alle Läufe): die Größe, in der die Auswahl der besten und schlechtesten Zuordnung geschieht."""
    mean_points = run.points.mean(axis=1)
    return float(mean_points.max() - mean_points.min())


def plan_lage(n: int, gap: float, runs: int, seed: int, m: float, rho: float, draw_max: float, plans: int, double: bool = False) -> Lage:
    """Bequemer Einstieg: Elo-Leiter, zufällige Zuordnungen (die erste ist die Identität), Simulation, beste gegen schlechteste Zuordnung.

    Ausgewählt wird nach den mittleren Punkten der stärksten Mannschaft an der ersten Hälfte der Läufe (ruhiger als die Titelchance), gemessen wird die Titelchance an der zweiten."""
    ratings = elo_ladder(n, gap)
    perms = random_assignments(n, plans, seed + 1)
    run = simulate_plans(ratings, perms, runs, seed, m, rho, draw_max, double=double)
    return split_best_worst(run.share, select=run.points)
