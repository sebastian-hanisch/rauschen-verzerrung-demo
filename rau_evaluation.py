"""Auswertung: die vier Modelle in einer gemeinsamen Einheit, die benötigte Zahl beobachteter Wettbewerbe, Messreihen, Rauschgrenze und Verdikte.

**Einheit (DEMO-PLAYBOOK §0, gleiche Zähleinheit):**
* Verzerrung: der **Lagenunterschied Δ**, die Chance der wahrhaft stärksten Seite zu gewinnen in der besten Lage minus in der schlechtesten Lage; Anteile, angezeigt in
  Prozentpunkten (Pp) dieser Chance. „Gewinnen“ heißt je Modell: Titel der Liga, Weiterkommen im Duell, höchste Wertung, Elfmeterschießen gewonnen.
* Beobachtung: **ein beobachteter Wettbewerb** (eine Saison, ein Duell, ein Auftritts-Wettbewerb, ein Elfmeterschießen), Ergebnis „die stärkste Seite gewinnt“ ja/nein.
  Die benötigte Zahl N zählt beobachtete Wettbewerbe **je Lage** (Messanordnung „zwei Lagen“) bzw. in der besten Lage (gegen den Fairwert, die Mitte der beiden Lagen).
"""

import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

import rau_constants as C
import rau_duel as D
import rau_judges as J
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_shootout as SO
from rau_lage import exact_lage

DATA_FILE = Path(__file__).resolve().parent / "data" / "messreihe.json"


# --- Anzeigehilfen (deutsche Schreibweise) ---------------------------------------------------------------------------------------------------------------

def de(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(p: float, digits: int = 1) -> str:
    """Anteil als Prozentzahl mit Dezimalkomma und Prozentzeichen."""
    return de(100.0 * p, digits) + " %"


def pp(delta: float, digits: int = 2) -> str:
    """Differenz zweier Anteile in Prozentpunkten (Pp), mit Vorzeichen, ohne „−0,00“."""
    text = de(100.0 * delta, digits)
    if float(text.replace(",", ".")) == 0.0:
        return de(0.0, digits) + " Pp"
    return ("+" if delta > 0 else "−") + text.lstrip("-") + " Pp"


def count(n: float) -> str:
    """Zahl der Wettbewerbe: ganze Zahl mit Tausenderpunkt; „unbestimmt“ bei unendlich."""
    if math.isinf(n):
        return "unbestimmt (Δ = 0)"
    return f"{int(round(n)):,}".replace(",", ".")


# --- Einstellungen ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Reglerwerte; immer mit Schlüsselwörtern aufrufen (ein eingefügtes Feld würde sonst still Werte verschieben)."""

    seed: int = C.DEFAULT_SEED
    runs: int = C.DEFAULT_RUNS
    alpha: int = C.DEFAULT_ALPHA            # Prozent
    power: int = C.DEFAULT_POWER            # Prozent
    design: str = C.DEFAULT_DESIGN
    example: str = C.DEFAULT_EXAMPLE
    plan_n: int = C.DEFAULT_PLAN_N
    plan_gap: int = C.DEFAULT_PLAN_GAP
    form_m: int = C.DEFAULT_FORM_M
    form_rho: float = C.DEFAULT_FORM_RHO
    plan_double: bool = C.DEFAULT_PLAN_DOUBLE
    lam_home: float = C.DEFAULT_LAM_HOME
    lam_away: float = C.DEFAULT_LAM_AWAY
    duel_elo: int = C.DEFAULT_DUEL_ELO
    away_rule: bool = C.DEFAULT_AWAY_RULE
    extra_time: bool = C.DEFAULT_EXTRA_TIME
    perf_n: int = C.DEFAULT_PERF_N
    judges: int = C.DEFAULT_JUDGES
    drift: float = C.DEFAULT_DRIFT
    anchor: float = C.DEFAULT_ANCHOR
    pen_p: float = C.DEFAULT_PEN_P
    pen_gap: float = C.DEFAULT_PEN_GAP
    pen_d: float = C.DEFAULT_PEN_D
    pen_order: str = C.DEFAULT_PEN_ORDER
    noise_n: int = C.DEFAULT_NOISE_N
    noise_games: int = C.DEFAULT_NOISE_GAMES

    @property
    def alpha_p(self) -> float:
        return self.alpha / 100.0

    @property
    def power_p(self) -> float:
        return self.power / 100.0


# --- Die vier Modelle als Zeilen der gemeinsamen Tabelle ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """Ein Modell in der gemeinsamen Einheit. Anteile, nicht Prozent; `se` ist nan bei exakter Rechnung."""

    model: str
    good: str
    bad: str
    p_good: float
    p_bad: float
    delta: float
    se: float
    exact: bool

    @property
    def p_fair(self) -> float:
        """Mitte der beiden Lagen: hier der Fairwert genannt; er ist nicht die Chance ohne Verzerrung (beim Auftritt bei den Voreinstellungen 76,7 % gegen 81,7 % ohne Drift)."""
        return (self.p_good + self.p_bad) / 2.0


def plan_row(n: int, gap: int, runs: int, seed: int, m: int, rho: float, double: bool) -> Row:
    lage = PL.plan_lage(n, float(gap), runs, seed, float(m), rho, C.PLAN_DRAW_MAX, C.PLAN_ASSIGNMENTS, double)
    return Row("Spielplan und Form", "beste Zuordnung", "schlechteste Zuordnung", lage.p_best, lage.p_worst, lage.delta, lage.se, False)


def duel_row(lam_home: float, lam_away: float, elo: int, rule: bool, extra_time: bool) -> Row:
    with_home, without_home, _ = D.lage_effect(lam_home, lam_away, float(elo), rule, extra_time, C.DUEL_PEN_STRONG)
    names = ("Rückspiel zu Hause", "Hinspiel zu Hause")
    lage = exact_lage([with_home, without_home])
    return Row("Hin- und Rückspiel", names[lage.best], names[lage.worst], lage.p_best, lage.p_worst, lage.delta, float("nan"), True)


def judge_row(n: int, judges: int, drift: float, anchor: float, runs: int, seed: int) -> Row:
    lage, _ = J.auftritt_lage(n, judges, C.JUDGE_SIGMA, drift, anchor, runs, seed)
    return Row("Auftrittsreihenfolge", f"Platz {lage.best + 1}", f"Platz {lage.worst + 1}", lage.p_best, lage.p_worst, lage.delta, lage.se, False)


def shoot_row(p: float, gap: float, d: float, order: str) -> Row:
    first, second, _ = SO.lage_effect(p, gap, d, order, C.PEN_SHOTS)
    names = ("schießt zuerst", "schießt als zweite")
    lage = exact_lage([first, second])
    return Row("Elfmeterschießen", names[lage.best], names[lage.worst], lage.p_best, lage.p_worst, lage.delta, float("nan"), True)


def analyse(s: Settings) -> list[Row]:
    """Alle vier Modelle bei den Reglerwerten `s`, in der Reihenfolge C.MODELS."""
    return [
        plan_row(s.plan_n, s.plan_gap, s.runs, s.seed, s.form_m, s.form_rho, s.plan_double),
        duel_row(s.lam_home, s.lam_away, s.duel_elo, s.away_rule, s.extra_time),
        judge_row(s.perf_n, s.judges, s.drift, s.anchor, s.runs, s.seed),
        shoot_row(s.pen_p, s.pen_gap, s.pen_d, s.pen_order),
    ]


# --- Benötigte Zahl beobachteter Wettbewerbe ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Needed:
    """Benötigte Wettbewerbe (ganze Zahlen, `inf` wenn kein Unterschied) mit der Spanne, die der Standardfehler von Δ zulässt (2 Standardfehler)."""

    n: float
    low: float       # bei Δ + 2 SE (weniger Wettbewerbe)
    high: float      # bei Δ - 2 SE (mehr Wettbewerbe; unendlich, wenn Δ - 2 SE nicht über null liegt)


def _shifted(row: Row, new_delta: float) -> Row:
    mid = row.p_fair
    return replace(row, p_good=mid + new_delta / 2.0, p_bad=mid - new_delta / 2.0, delta=new_delta)


def needed_for(row: Row, alpha: float, power: float, design: str) -> float:
    """Benötigte Wettbewerbe (aufgerundet) für die Zeile; bei Δ <= 0 unbestimmt (die beste Lage ist dann nicht besser als die schlechteste)."""
    if row.delta <= 0.0:
        return math.inf
    return PW.whole_competitions(PW.n_needed(row.p_good, row.p_bad, alpha, power, design))


def needed(row: Row, alpha: float, power: float, design: str) -> Needed:
    n = needed_for(row, alpha, power, design)
    if row.exact or math.isnan(row.se):
        return Needed(n, n, n)
    return Needed(n, needed_for(_shifted(row, row.delta + 2.0 * row.se), alpha, power, design),
                  needed_for(_shifted(row, row.delta - 2.0 * row.se), alpha, power, design))


def table_rows(rows: list[Row], s: Settings) -> list[dict]:
    """Zeilen der Vergleichstabelle als Wörterbücher mit Rohwerten (die App formatiert)."""
    out = []
    for r in rows:
        nd = needed(r, s.alpha_p, s.power_p, s.design)
        out.append({"model": r.model, "good": r.good, "bad": r.bad, "p_good": r.p_good, "p_bad": r.p_bad, "p_fair": r.p_fair, "noise": 1.0 - r.p_fair,
                    "delta": r.delta, "se": r.se, "exact": r.exact, "n": nd.n, "n_low": nd.low, "n_high": nd.high})
    return out


# --- Wachsendes Beispiel -------------------------------------------------------------------------------------------------------------------------------

def example_row(rows: list[Row], model: str) -> Row:
    for r in rows:
        if r.model == model:
            return r
    raise ValueError(f"Modell {model!r} unbekannt")


def example_horizon(n_two: float) -> int:
    """Zahl der gezeigten Saisons: etwa das Dreifache der benötigten Zahl, mindestens 60 und höchstens C.EXAMPLE_MAX_SEASONS."""
    if math.isinf(n_two):
        return C.EXAMPLE_MAX_SEASONS
    return int(min(C.EXAMPLE_MAX_SEASONS, max(60, math.ceil(3 * n_two))))


def example_path(row: Row, s: Settings) -> tuple[PW.SeasonPath, float]:
    """Ein gewürfeltes Beispiel für die Zeile: in jeder Saison ein Wettbewerb in der guten und einer in der schlechten Lage (Zufalls-Seed der Seitenleiste).
    Gibt den Weg und die nach der geschlossenen Form benötigte Zahl (zwei Lagen) zurück."""
    n_two = needed_for(row, s.alpha_p, s.power_p, "zwei Lagen")
    p_bad = row.p_bad if row.delta > 0 else row.p_good
    path = PW.season_path(row.p_good, p_bad, example_horizon(n_two), s.alpha_p, s.seed)
    return path, n_two


# --- Rauschgrenze ----------------------------------------------------------------------------------------------------------------------------------------

def noise_rows(n: int, seed: int, runs: int = C.NOISE_RUNS) -> list[dict]:
    """Chance, dass die Stärkste an der Spitze steht, für die Raster der Partien je Paar (Elo-Leiter mit 0,3 logistischer Stufe, ohne Remis)."""
    results = NO.noise_curve(n, C.NOISE_GAMES_OPTIONS, NO.ELO_STEP, runs, seed)
    return [{"games": g, "p_top": r.p_top, "se": r.se} for g, r in zip(C.NOISE_GAMES_OPTIONS, results)]


# --- Vorgerechnete Messreihen -------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_messreihe() -> dict:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _mean(values) -> float:
    return float(np.mean(values))


def sweep_plan(rho: float) -> list[dict]:
    """Spielplan, Wirkung der Form m = 0 ... 150 bei Gedächtnis ρ: Mittel über die fünf Zufalls-Seeds der Messreihe (Standardwerte der übrigen Regler)."""
    data = load_messreihe()["plan"]
    rows = []
    for m in C.SWEEP_FORM_M:
        cell = data[f"m={m}|rho={rho}"]
        rows.append({"x": m, "p_good": _mean(cell["p_good"]), "p_bad": _mean(cell["p_bad"]), "delta": _mean(cell["delta"]), "spread": float(np.std(cell["delta"], ddof=1))})
    return rows


def sweep_judges_drift() -> list[dict]:
    data = load_messreihe()["auftritt_drift"]
    return [{"x": b, "p_good": _mean(data[f"{b}"]["p_good"]), "p_bad": _mean(data[f"{b}"]["p_bad"]), "delta": _mean(data[f"{b}"]["delta"]),
             "spread": float(np.std(data[f"{b}"]["delta"], ddof=1))} for b in C.SWEEP_DRIFT]


def sweep_judges_anchor() -> list[dict]:
    data = load_messreihe()["auftritt_anchor"]
    return [{"x": g, "p_good": _mean(data[f"{g}"]["p_good"]), "p_bad": _mean(data[f"{g}"]["p_bad"]), "delta": _mean(data[f"{g}"]["delta"]),
             "spread": float(np.std(data[f"{g}"]["delta"], ddof=1))} for g in C.SWEEP_ANCHOR]


def sweep_duel(rule: bool, elo: int = C.DEFAULT_DUEL_ELO, extra_time: bool = True) -> list[dict]:
    """Hin- und Rückspiel, Heim-Mittel 1,3 ... 2,0 bei Summe 2,6 (exakt)."""
    rows = []
    for lh in C.SWEEP_LAM_HOME:
        r = duel_row(lh, round(C.SWEEP_LAM_TOTAL - lh, 10), elo, rule, extra_time)
        rows.append({"x": lh, "p_good": r.p_good, "p_bad": r.p_bad, "delta": r.delta, "spread": float("nan")})
    return rows


def sweep_shoot(order: str, p: float = C.DEFAULT_PEN_P, gap: float = C.DEFAULT_PEN_GAP) -> list[dict]:
    """Elfmeterschießen, Druck d = 0 ... 0,25 (exakt)."""
    rows = []
    for d in C.SWEEP_PEN_D:
        r = shoot_row(p, gap, d, order)
        rows.append({"x": d, "p_good": r.p_good, "p_bad": r.p_bad, "delta": r.delta, "spread": float("nan")})
    return rows


def sweep_needed(rows: list[dict], alpha: float, power: float, design: str) -> list[float]:
    """Benötigte Wettbewerbe zu jeder Zeile einer Messreihe (unendlich bei Δ <= 0)."""
    out = []
    for r in rows:
        row = Row("", "", "", r["p_good"], r["p_bad"], r["delta"], float("nan"), True)
        out.append(needed_for(row, alpha, power, design))
    return out


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------------------

def verdict_comparison(table: list[dict]) -> tuple[str, str]:
    """Verdikt zum Vergleich der vier Modelle: kleinste und größte benötigte Zahl, mit der Einheit; bei Δ = 0 in allen Modellen die Rauschen-Meldung."""
    finite = [r for r in table if not math.isinf(r["n"])]
    if not finite:
        return "info", "In allen vier Modellen ist der Lagenunterschied Δ null (oder negativ gemessen): Es gibt nichts zu sehen, was über das Rauschen hinausgeht. Die benötigte Zahl beobachteter Wettbewerbe ist unbestimmt."
    easy = min(finite, key=lambda r: r["n"])
    hard = max(finite, key=lambda r: r["n"])
    text = (f"Einheit: Δ ist der Unterschied der Chance der stärksten Seite zu gewinnen zwischen bester und schlechtester Lage, in Prozentpunkten (Pp); N zählt beobachtete Wettbewerbe. "
            f"Am leichtesten fällt **{easy['model']}** auf (Δ = {pp(easy['delta'])}, N = {count(easy['n'])}), am schwersten **{hard['model']}** (Δ = {pp(hard['delta'])}, N = {count(hard['n'])}).")
    missing = [r["model"] for r in table if math.isinf(r["n"])]
    if missing:
        text += " Ohne Unterschied (N unbestimmt): " + ", ".join(missing) + "."
    undecided = [r["model"] for r in table if not r["exact"] and not math.isinf(r["n"]) and math.isinf(r["n_high"])]
    if undecided:
        text += " Der Standardfehler lässt bei " + ", ".join(undecided) + " zu, dass Δ null ist (Δ − 2 SE ≤ 0): dort ist N nach oben offen."
    if any(r["model"] == "Spielplan und Form" and not r["exact"] and not math.isinf(r["n"]) for r in table):
        text += (" Beim Spielplan ist Δ eine Einzelstichprobe: Die Streuung zwischen Zufalls-Seeds ist größer als der angezeigte Standardfehler "
                 "(bei den Voreinstellungen und 8.000 Läufen 0,98 Pp gegen 0,54 Pp, Zufalls-Seeds 101 bis 105), die Spanne für N ist dort zu eng.")
    ratio = hard["n"] / easy["n"] if easy["n"] > 0 else math.inf
    kind = "success" if ratio >= 10 else "info"
    return kind, text + f" Das Verhältnis der beiden Zahlen ist {de(ratio, 1)}."


def verdict_example(row_model: str, path: PW.SeasonPath, n_two: float, horizon: int, alpha: float) -> tuple[str, str]:
    """Verdikt zum wachsenden Beispiel: erste Saison außerhalb des Zufallsbandes und die Zahl der Formel."""
    level = de(100.0 * alpha, 0) + " %"
    if math.isinf(n_two):
        return "info", f"{row_model}: Es gibt keinen Lagenunterschied; die Linie bleibt im Zufallsband (Fehlerniveau {level}), und die Zahl der Saisons ist unbestimmt."
    base = f"{row_model}: Die Formel verlangt N = {count(n_two)} Wettbewerbe je Lage (im Beispiel je „Saison“ ein Wettbewerb je Lage). In diesem gewürfelten Beispiel ({horizon} Saisons gezeigt) "
    if path.first_seen == 0:
        return "warning", base + "verlässt die Linie das Zufallsband nicht."
    tail = f"verlässt die Linie das Band erstmals in Saison {path.first_seen}"
    if path.stays_out:
        tail += f" und bleibt ab Saison {path.stays_out} draußen."
    else:
        tail += ", liegt aber am Ende wieder im Band."
    return "success", base + tail


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zur Rauschgrenze: Chance der Stärksten an der Spitze bei der kleinsten und größten Zahl von Partien je Paar."""
    first, last = rows[0], rows[-1]
    return "info", (f"Auch ohne jede Verzerrung steht die stärkste Mannschaft bei {first['games']} Partie(n) je Paar nur in {pct(first['p_top'])} der Ligen an der Spitze "
                    f"(Standardfehler {de(100 * first['se'], 1)} Pp), bei {last['games']} Partien je Paar in {pct(last['p_top'])}. Mehr Partien verkleinern das Rauschen.")


def verdict_power(sim: PW.PowerSim, target: float, exact: float | None) -> tuple[str, str]:
    """Verdikt zur Probe: gewürfelte Teststärke gegen das Ziel und (falls gerechnet) gegen die exakte Binomialrechnung."""
    reps = f"{sim.reps:,}".replace(",", ".")
    text = f"Gewürfelt lehnt der Test in {pct(sim.rate)} von {reps}" + f" Wiederholungen ab (Standardfehler {de(100 * sim.se, 1)} Pp); Ziel {pct(target, 0)}."
    if exact is not None:
        text += f" Exakt über beide Binomialverteilungen: {pct(exact)}."
    close = abs(sim.rate - target) <= 0.05
    return ("success" if close else "warning"), text + (" Die Formel trifft das Ziel innerhalb von 5 Pp." if close else " Die Formel weicht um mehr als 5 Pp vom Ziel ab (Normalnäherung bei kleinen Zahlen).")


def verdict_sweep(series: dict[str, list[tuple[float, float]]]) -> tuple[str, str]:
    """Verdikt zur Messreihe über die Verzerrungsstärke: Spanne der benötigten Zahlen über alle gezeigten Punkte mit Δ > 0 (Reihen: Name -> [(Δ in Anteil, N)])."""
    points = [(name, d, n) for name, rows in series.items() for d, n in rows if d > 0 and not math.isinf(n)]
    if not points:
        return "info", "Keine Reihe hat einen Lagenunterschied über null."
    lo = min(points, key=lambda t: t[2])
    hi = max(points, key=lambda t: t[2])
    return "info", (f"Über alle gezeigten Punkte reicht die benötigte Zahl von {count(lo[2])} ({lo[0]}, Δ = {pp(lo[1])}) bis {count(hi[2])} ({hi[0]}, Δ = {pp(hi[1])}). "
                    "Je kleiner Δ, desto schneller wächst N: etwa mit 1/Δ².")
