"""Modell 4: Elfmeterschießen mit Druck. Gemessen wird, wie sehr es der **stärkeren** Mannschaft nützt oder schadet, zuerst zu schießen.

Exakte Rechnung: die Regelzeit (R Schüsse je Mannschaft) als Verteilung über die Torstände, danach Sudden Death in geschlossener Form. Ein Schütze trifft mit
der Wahrscheinlichkeit seiner Mannschaft p, im Rückstand mit p - d (Druck, Annahme); ein vorzeitiger Abbruch ändert den Sieger nie und wird nicht gerechnet.
Reihenfolge ABAB (die erste Mannschaft schießt in jedem Paar zuerst) oder ABBA (die Paare wechseln: AB, BA, AB, ...; das Paar R + 1 setzt den Wechsel fort).
Eigene, kompakte Fassung; sie importiert nichts aus anderen Demos.
"""

ORDERS = ("ABAB", "ABBA")


def _shot_probability(p: float, d: float, behind: bool) -> float:
    return max(0.0, p - d) if behind else p


def regulation_distribution(p_first: float, p_second: float, d: float, order: str, shots: int) -> dict:
    """Verteilung der Torstände (Tore der ersten, Tore der zweiten Mannschaft) nach `shots` Schüssen je Mannschaft in der Regelzeit."""
    if order not in ORDERS:
        raise ValueError(f"Reihenfolge {order!r} unbekannt")
    if shots < 1:
        raise ValueError("Mindestens ein Schuss je Mannschaft")
    state = {(0, 0): 1.0}
    for pair in range(shots):
        a_first = order == "ABAB" or pair % 2 == 0
        for team in ((0, 1) if a_first else (1, 0)):
            nxt: dict = {}
            for (ga, gb), mass in state.items():
                own, other = (ga, gb) if team == 0 else (gb, ga)
                p = _shot_probability(p_first if team == 0 else p_second, d, own < other)
                for hit, weight in ((1, p), (0, 1.0 - p)):
                    if weight == 0.0:
                        continue
                    key = (ga + hit, gb) if team == 0 else (ga, gb + hit)
                    nxt[key] = nxt.get(key, 0.0) + mass * weight
            state = nxt
    return state


def sudden_death_first_wins(p_first: float, p_second: float, d: float, order: str, shots: int) -> float:
    """Chance der ersten Mannschaft (A), den Sudden Death nach Gleichstand zu gewinnen (geschlossene Form).

    Paar mit A zuerst: A trifft mit p_A; B schießt im Rückstand (p_B - d), wenn A traf. A gewinnt das Paar mit p_A·(1 - (p_B - d)), B mit (1 - p_A)·p_B.
    Paar mit B zuerst entsprechend. Bei ABAB beginnt A immer; bei ABBA beginnt das Paar mit der Nummer `shots` (von 0 gezählt) mit A, wenn `shots` gerade ist."""
    pa, pb = p_first, p_second
    a1 = pa * (1.0 - max(0.0, pb - d))
    b1 = (1.0 - pa) * pb
    c1 = 1.0 - a1 - b1
    if order == "ABAB":
        return 0.5 if a1 + b1 < 1e-15 else a1 / (a1 + b1)
    a2 = (1.0 - pb) * pa
    b2 = pb * (1.0 - max(0.0, pa - d))
    c2 = 1.0 - a2 - b2
    denom = 1.0 - c1 * c2
    if denom < 1e-15:
        return 0.5
    y = (a1 + c1 * a2) / denom        # A beginnt das Paar
    x = a2 + c2 * y                   # B beginnt das Paar
    return y if shots % 2 == 0 else x


def first_team_win(p_first: float, p_second: float, d: float, order: str = "ABAB", shots: int = 5) -> float:
    """Chance der Mannschaft, die zuerst schießt, das Elfmeterschießen zu gewinnen."""
    dist = regulation_distribution(p_first, p_second, d, order, shots)
    win = sum(mass for (ga, gb), mass in dist.items() if ga > gb)
    tie = sum(mass for (ga, gb), mass in dist.items() if ga == gb)
    return win + tie * sudden_death_first_wins(p_first, p_second, d, order, shots)


def lage_effect(p_strong: float, gap: float, d: float, order: str = "ABAB", shots: int = 5):
    """(Chance der stärkeren Mannschaft, wenn sie zuerst schießt, wenn sie als zweite schießt) und ihre Differenz; die schwächere trifft mit p_strong - gap."""
    if gap < 0:
        raise ValueError("Die stärkere Mannschaft hat die höhere Trefferwahrscheinlichkeit (gap >= 0)")
    p_weak = p_strong - gap
    if not 0.0 <= p_weak <= 1.0 or not 0.0 <= p_strong <= 1.0:
        raise ValueError("Trefferwahrscheinlichkeiten müssen in [0, 1] liegen")
    good = first_team_win(p_strong, p_weak, d, order, shots)
    bad = 1.0 - first_team_win(p_weak, p_strong, d, order, shots)
    return good, bad, good - bad
