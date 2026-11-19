"""Rechnet die Messreihen vor, die die App als Datei liest (data/messreihe.json).

Spielplan-Modell: Wirkung der Form m (Raster C.SWEEP_FORM_M) bei Gedächtnis ρ in C.SWEEP_FORM_RHO; Auftritts-Modell: Drift (C.SWEEP_DRIFT) und Anker (C.SWEEP_ANCHOR);
je Zelle fünf feste Zufalls-Seeds (C.SWEEP_SEEDS) mit je C.SWEEP_RUNS Wettbewerben. Alle anderen Regler stehen auf den Standardwerten. Aufruf (Demo-Ordner als Arbeitsverzeichnis):

    C:/Users/sebas/Claude Code Projects/_venvs/test/Scripts/python.exe tools/make_messreihe.py

Laufzeit: rund 1 bis 4 Minuten auf mehreren Prozessen. Das Ergebnis ist bei gleichen Bibliotheksversionen bitgleich reproduzierbar; unter anderen Versionen
oder Plattformen kann sich die letzte Stelle ändern (die Tests prüfen mit Band).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rau_constants as C  # noqa: E402
import rau_evaluation as EV  # noqa: E402


def plan_cell(args):
    m, rho, seed = args
    r = EV.plan_row(C.DEFAULT_PLAN_N, C.DEFAULT_PLAN_GAP, C.SWEEP_RUNS, seed, m, rho, C.DEFAULT_PLAN_DOUBLE)
    return ("plan", f"m={m}|rho={rho}", seed, r)


def judge_cell(args):
    kind, x, seed = args
    drift, anchor = (x, C.DEFAULT_ANCHOR) if kind == "auftritt_drift" else (C.DEFAULT_DRIFT, x)
    r = EV.judge_row(C.DEFAULT_PERF_N, C.DEFAULT_JUDGES, drift, anchor, C.SWEEP_RUNS, seed)
    return (kind, f"{x}", seed, r)


def main() -> None:
    jobs_plan = [(m, rho, seed) for m in C.SWEEP_FORM_M for rho in C.SWEEP_FORM_RHO for seed in C.SWEEP_SEEDS]
    jobs_judge = [("auftritt_drift", b, seed) for b in C.SWEEP_DRIFT for seed in C.SWEEP_SEEDS] + [("auftritt_anchor", g, seed) for g in C.SWEEP_ANCHOR for seed in C.SWEEP_SEEDS]
    out: dict = {"meta": {"runs": C.SWEEP_RUNS, "seeds": list(C.SWEEP_SEEDS), "plan_n": C.DEFAULT_PLAN_N, "plan_gap": C.DEFAULT_PLAN_GAP, "perf_n": C.DEFAULT_PERF_N,
                          "judges": C.DEFAULT_JUDGES, "assignments": C.PLAN_ASSIGNMENTS}, "plan": {}, "auftritt_drift": {}, "auftritt_anchor": {}}
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(plan_cell, jobs_plan)) + list(pool.map(judge_cell, jobs_judge))
    for kind, key, seed, r in results:
        cell = out[kind].setdefault(key, {"seed": [], "p_good": [], "p_bad": [], "delta": [], "se": []})
        cell["seed"].append(seed)
        cell["p_good"].append(round(r.p_good, 6))
        cell["p_bad"].append(round(r.p_bad, 6))
        cell["delta"].append(round(r.delta, 6))
        cell["se"].append(round(r.se, 6))
    target = ROOT / "data" / "messreihe.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"geschrieben: {target} ({len(results)} Zellen-Läufe)")


if __name__ == "__main__":
    main()
