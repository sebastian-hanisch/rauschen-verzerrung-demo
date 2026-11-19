"""Rauschen oder Verzerrung? Wie viele Saisons braucht man, um sie zu unterscheiden? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zusammenfluss der Linie „Wettbewerbsverzerrung erklärt“: vier Verzerrungen (Spielplan und Form, Hin- und Rückspiel, Auftrittsreihenfolge, Elfmeterschießen) in
**einer** Einheit, dem Lagenunterschied Δ (Prozentpunkte der Gewinnchance der stärksten Seite), und die Zahl der beobachteten Wettbewerbe, die man braucht, um Δ gegen
den Zufall zu sehen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import rau_constants as C
import rau_evaluation as EV
import rau_power as PW
from rau_layout import dunkles_design, kopf_css
from rau_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rau_visualization import build_compare, build_example, build_needed, build_noise, build_sweep

st.set_page_config(page_title="Rauschen oder Verzerrung? Wie viele Saisons braucht man, um sie zu unterscheiden? – Sebastian Hanisch", layout="wide")


# --- Berechnungen, je Modell einzeln gepuffert (der Schlüssel enthält alle Eingaben, auch den Zufalls-Seed) --------------------------------------------------

@st.cache_data(show_spinner=False)
def _plan(n, gap, runs, seed, m, rho, double):
    return EV.plan_row(n, gap, runs, seed, m, rho, double)


@st.cache_data(show_spinner=False)
def _duel(lam_home, lam_away, elo, rule, extra_time):
    return EV.duel_row(lam_home, lam_away, elo, rule, extra_time)


@st.cache_data(show_spinner=False)
def _judges(n, judges, drift, anchor, runs, seed):
    return EV.judge_row(n, judges, drift, anchor, runs, seed)


@st.cache_data(show_spinner=False)
def _shoot(p, gap, d, order):
    return EV.shoot_row(p, gap, d, order)


@st.cache_data(show_spinner=False)
def _noise(n, seed):
    return EV.noise_rows(n, seed)


@st.cache_data(show_spinner=False)
def _sweeps():
    series = {
        "Spielplan, ρ = 0": ("plan", EV.sweep_plan(0.0)),
        "Spielplan, ρ = 0,5": ("plan", EV.sweep_plan(0.5)),
        "Spielplan, ρ = 0,8": ("plan", EV.sweep_plan(0.8)),
        "Rückspiel ohne Auswärtstorregel": ("duel", EV.sweep_duel(False)),
        "Rückspiel mit Auswärtstorregel": ("duel", EV.sweep_duel(True)),
        "Auftritt: Drift": ("judge", EV.sweep_judges_drift()),
        "Auftritt: Angleichung oder Kontrast": ("judge", EV.sweep_judges_anchor()),
        "Elfmeter ABAB": ("shoot", EV.sweep_shoot("ABAB")),
        "Elfmeter ABBA": ("shoot", EV.sweep_shoot("ABBA")),
    }
    return series


@st.cache_data(show_spinner=False)
def _peeking(p, seasons, alpha, seed):
    return PW.peeking_rate(p, seasons, alpha, C.PEEK_PATHS, seed)


@st.cache_data(show_spinner=False)
def _power(p_good, p_bad, n, alpha, design, seed):
    exact = PW.exact_power(p_good, p_bad, n, alpha, design) if n <= PW.EXACT_N_LIMIT else None
    return PW.simulate_power(p_good, p_bad, n, alpha, C.POWER_REPS, seed, design), exact


def _fmt_table(table):
    rows = []
    for r in table:
        span = "—"
        if not r["exact"] and not math.isinf(r["n"]):
            high = "offen" if math.isinf(r["n_high"]) else EV.count(r["n_high"])
            span = f"{EV.count(r['n_low'])} bis {high}"
        rows.append({
            "Modell": r["model"],
            "Beste Lage": r["good"],
            "Schlechteste Lage": r["bad"],
            "Chance in der besten Lage": EV.pct(r["p_good"]),
            "Chance in der schlechtesten Lage": EV.pct(r["p_bad"]),
            "Lagenunterschied Δ": EV.pp(r["delta"]),
            "Fairwert (Mitte der beiden Lagen)": EV.pct(r["p_fair"]),
            "Benötigte Wettbewerbe N": EV.count(r["n"]),
            "N bei Δ ± 2 Standardfehler": span,
            "Rechnung": "exakt" if r["exact"] else "Simulation",
        })
    return pd.DataFrame(rows)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rau_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Rauschen oder Verzerrung? Wie viele Saisons braucht man, um sie zu unterscheiden?")
st.markdown(
    """
Eine **Verzerrung** verschiebt Ergebnisse: wer im Rückspiel zu Hause spielt, wer zuletzt auftritt, in welcher Reihenfolge ein Spielplan die Gegner stellt, wer beim Elfmeterschießen beginnt. Der **Zufall** steckt ohnehin in jedem Wettbewerb. Diese Demo stellt vier
Verzerrungen in **einer gemeinsamen Einheit** nebeneinander, dem **Lagenunterschied Δ** (Prozentpunkte der Chance, dass die wahrhaft stärkste Seite gewinnt, beste gegen schlechteste Lage), und rechnet aus, **wie viele beobachtete Wettbewerbe** man braucht, um Δ gegen den Zufall zu sehen.
Alles ist **synthetisch** (keine realen Vereine, keine Felddaten); die Ergebnisse gelten für die Annahmen der vier Modelle.

Weiter unten: die **Kernfrage**, vier **🔬 Experimente** (wachsendes Beispiel, Rauschgrenze, Probe der Teststärke, Messreihe über die Verzerrungsstärke), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Messung", expanded=True):
    st.markdown(
        """
1. **Lage und Lagenunterschied Δ.** Eine *Lage* ist eine Ausgangsposition, die mit der Stärke nichts zu tun hat: das Rückspiel zu Hause, die Zuordnung im Spielplan, die Auftrittsnummer, die Schussreihenfolge. Δ ist die Chance der **wahrhaft stärksten** Seite, zu gewinnen, in der besten Lage **minus** dieselbe Chance in der schlechtesten Lage, in
   Prozentpunkten (Pp). „Gewinnen“ heißt: die Liga gewinnen, im Duell weiterkommen, die höchste Wertung bekommen, das Elfmeterschießen gewinnen. Die vier Modelle sind in dieser Demo je eine eigene, kompakte Rechnung mit bekannter wahrer Stärke.
2. **Ein beobachteter Wettbewerb liefert nur ja oder nein** (die Stärkste gewinnt oder nicht). Das ist das **Rauschen**: bei der Chance p hat jede Beobachtung die Varianz p·(1 − p). Um einen Unterschied Δ zwischen zwei Lagen zu sehen, muss man so viele Wettbewerbe je Lage beobachten, dass der Unterschied der Anteile über dem Zufallsrauschen liegt.
3. **Die benötigte Zahl N** hängt vom Fehlerniveau (Fehlalarme, hier zweiseitig) und von der **Teststärke** ab (Chance, einen echten Unterschied zu finden). Messanordnung **„zwei Lagen“**: je N Wettbewerbe in der besten und in der schlechtesten Lage, geprüft wird der Unterschied der Anteile. Messanordnung **„gegen den Fairwert“**: N Wettbewerbe in der besten Lage, geprüft
   wird der Anteil gegen den als bekannt angenommenen Fairwert (die Mitte beider Lagen, nicht die Chance ohne Verzerrung); das setzt voraus, dass man den Fairwert kennt. Gegen den Fairwert braucht man in der besten Lage etwa doppelt so viele Wettbewerbe wie je Lage beim Vergleich zweier Lagen.
4. **Der Hot-Hand-Messfehler steht nicht in der Tabelle.** Dort bleibt die Gewinnchance unverändert: dieselbe Zufallsfolge wird mit einem anderen Maß ausgewertet, und das gängige Maß liegt dabei im Mittel neben dem wahren Wert (so im Stück der Linie zum Hot-Hand-Messfehler gemessen, hier nicht nachgerechnet). Das ist ein Problem der Messung, keine Verschiebung der Gewinnchance einer Lage; es hätte in dieser Einheit keinen Lagenunterschied (Δ wäre null).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

    st.header("⚙️ Messung und Test")
    st.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="seed", step=1, help="Gleiche Reglerwerte und gleicher Zufalls-Seed ergeben dasselbe Ergebnis.")
    st.select_slider("Gewürfelte Wettbewerbe je Messung", C.RUNS_OPTIONS, key="runs",
                     help="Gilt für die Modelle Spielplan und Auftrittsreihenfolge (Simulation). Hin- und Rückspiel und Elfmeterschießen werden exakt gerechnet. Mehr Läufe verkleinern den Standardfehler von Δ und rechnen länger.")
    st.select_slider("Fehlerniveau α (zweiseitig)", C.ALPHA_OPTIONS, key="alpha", format_func=lambda v: f"{v} %",
                     help="Wie oft der Test einen Unterschied meldet, obwohl keiner da ist (Fehlalarm).")
    st.select_slider("Gewünschte Teststärke", C.POWER_OPTIONS, key="power", format_func=lambda v: f"{v} %",
                     help="Wie oft der Test einen echten Unterschied der Größe Δ finden soll.")
    st.radio("Messanordnung", C.DESIGNS, key="design",
             help="„zwei Lagen“: je N Wettbewerbe in beiden Lagen, geprüft wird der Unterschied. „gegen den Fairwert“: N Wettbewerbe in der besten Lage gegen den bekannten Fairwert (die Mitte der beiden Lagen).")
    st.selectbox("Modell für das wachsende Beispiel", C.MODELS, key="example", help="Welche Verzerrung das Experiment „Saison für Saison“ und die Probe der Teststärke zeigen.")

    st.header("🗓️ Spielplan und Form")
    st.select_slider("Mannschaften", C.PLAN_N_OPTIONS, key="plan_n", help="Liga nach der Zirkelmethode; bei 4 Mannschaften sind alle 24 Zuordnungen dabei, sonst 24 zufällige.")
    st.slider("Elo-Abstand Stärkster–Schwächster", C.PLAN_GAP_MIN, C.PLAN_GAP_MAX, key="plan_gap", step=C.PLAN_GAP_STEP, help="Stärken gleichmäßig auf einer Elo-Leiter.")
    st.slider("Wirkung der Form m (Elo-Punkte)", C.FORM_M_MIN, C.FORM_M_MAX, key="form_m", step=C.FORM_M_STEP,
              help="Nach einem Sieg ist eine Mannschaft um m Elo-Punkte stärker, nach einer Niederlage um m schwächer (Zustand mit Gedächtnis). Bei 0 gibt es keine Verzerrung durch den Spielplan: jede Zuordnung ergibt dasselbe.")
    st.slider("Gedächtnis ρ des Zustands", C.FORM_RHO_MIN, C.FORM_RHO_MAX, key="form_rho", step=C.FORM_RHO_STEP,
              help="Wahrscheinlichkeit, dass der Zustand nach einer Partie bleibt statt auf das Ergebnis gesetzt zu werden.")
    st.toggle("Doppelrunde (Hin- und Rückrunde)", key="plan_double", help="Die Rückrunde wiederholt die Hinrunde in gleicher Reihenfolge; der Zustand läuft durch.")

    st.header("🔁 Hin- und Rückspiel")
    st.slider("Heim-Mittel (Tore je 90 Minuten)", C.LAM_HOME_MIN, C.LAM_HOME_MAX, key="lam_home", step=C.LAM_STEP, help="Mittlere Tore des Gastgebers bei gleich starken Teams (Annahme).")
    st.slider("Auswärts-Mittel (Tore je 90 Minuten)", C.LAM_AWAY_MIN, C.LAM_AWAY_MAX, key="lam_away", step=C.LAM_STEP, help="Mittlere Tore des Gastes bei gleich starken Teams (Annahme).")
    st.slider("Elo-Abstand der stärkeren Mannschaft", C.DUEL_ELO_MIN, C.DUEL_ELO_MAX, key="duel_elo", step=C.DUEL_ELO_STEP,
              help="Die stärkere Seite schießt Tore mit dem Faktor 10^(d/800), die schwächere mit 10^(−d/800).")
    st.toggle("Auswärtstorregel (Modell)", key="away_rule", help="Bei gleicher Tor-Summe gewinnt, wer mehr Auswärtstore hat; eine frühere Regel, hier nur als Modell.")
    st.toggle("Verlängerung", key="extra_time", help="Bei Gleichstand 30 Minuten im Stadion des Rückspiel-Gastgebers mit einem Drittel der Torrate; ohne Verlängerung folgt sofort das Elfmeterschießen (50 : 50).")

    st.header("🎭 Auftrittsreihenfolge")
    st.slider("Auftretende", C.PERF_N_MIN, C.PERF_N_MAX, key="perf_n", help="Zahl der Auftretenden mit wahrer Qualität Normal(0, 1).")
    st.slider("Kampfrichter", C.JUDGES_MIN, C.JUDGES_MAX, key="judges", help="Zahl der Kampfrichter; die Wertung ist das Mittel ihrer Noten (Rauschen 0,5 je Note).")
    st.slider("Drift je Platz (Notenpunkte)", C.DRIFT_MIN, C.DRIFT_MAX, key="drift", step=C.DRIFT_STEP, help="Spätere Plätze bekommen höhere Noten (bei positiver Drift) bei gleicher Qualität.")
    st.slider("Angleichung (+) oder Kontrast (−)", C.ANCHOR_MIN, C.ANCHOR_MAX, key="anchor", step=C.ANCHOR_STEP, help="Anteil der vorherigen Panelnote, der in die nächste Note eingeht (positiv: Angleichung, negativ: Kontrast).")

    st.header("🥅 Elfmeterschießen")
    st.slider("Trefferchance der stärkeren Mannschaft", C.PEN_P_MIN, C.PEN_P_MAX, key="pen_p", step=C.PEN_P_STEP, help="Trefferwahrscheinlichkeit je Schuss, wenn die Mannschaft nicht zurückliegt.")
    st.slider("Vorsprung der stärkeren Mannschaft", C.PEN_GAP_MIN, C.PEN_GAP_MAX, key="pen_gap", step=C.PEN_GAP_STEP, help="Die schwächere Mannschaft trifft mit der Chance der stärkeren minus diesem Vorsprung.")
    st.slider("Druck d im Rückstand", C.PEN_D_MIN, C.PEN_D_MAX, key="pen_d", step=C.PEN_D_STEP,
              help="Wer vor dem Schuss zurückliegt, trifft mit p − d. Bei d = 0 ist jeder Schuss unabhängig und die Reihenfolge ändert nichts.")
    st.radio("Reihenfolge", C.PEN_ORDERS, key="pen_order", horizontal=True,
             help="ABAB: die erste Mannschaft schießt in jedem Paar zuerst. ABBA: die Paare wechseln (AB, BA, AB, ...).")

    st.header("📉 Rauschgrenze")
    st.slider("Teilnehmer der Liga", C.NOISE_N_MIN, C.NOISE_N_MAX, key="noise_n", help="Fairer Vergleichsmaßstab: eine Liga ohne jede Verzerrung, k Partien je Paar, ohne Remis.")
    st.select_slider("Hervorgehobene Partien je Paar", C.NOISE_GAMES_OPTIONS, key="noise_games", help="Welcher Wert im Diagramm der Rauschgrenze hervorgehoben ist; gerechnet werden immer alle sechs Werte.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

ss = st.session_state
sync_query_params({key: ss[key] for key in SETTING_SPECS})

settings = EV.Settings(
    seed=int(ss.seed), runs=int(ss.runs), alpha=int(ss.alpha), power=int(ss.power), design=ss.design, example=ss.example,
    plan_n=int(ss.plan_n), plan_gap=int(ss.plan_gap), form_m=int(ss.form_m), form_rho=float(ss.form_rho), plan_double=bool(ss.plan_double),
    lam_home=float(ss.lam_home), lam_away=float(ss.lam_away), duel_elo=int(ss.duel_elo), away_rule=bool(ss.away_rule), extra_time=bool(ss.extra_time),
    perf_n=int(ss.perf_n), judges=int(ss.judges), drift=float(ss.drift), anchor=float(ss.anchor),
    pen_p=float(ss.pen_p), pen_gap=float(ss.pen_gap), pen_d=float(ss.pen_d), pen_order=ss.pen_order,
    noise_n=int(ss.noise_n), noise_games=int(ss.noise_games),
)

with st.spinner("Rechnet die Simulationen (Spielplan und Auftritt, einige Sekunden bei neuen Reglerwerten) …"):
    rows = [
        _plan(settings.plan_n, settings.plan_gap, settings.runs, settings.seed, settings.form_m, settings.form_rho, settings.plan_double),
        _duel(settings.lam_home, settings.lam_away, settings.duel_elo, settings.away_rule, settings.extra_time),
        _judges(settings.perf_n, settings.judges, settings.drift, settings.anchor, settings.runs, settings.seed),
        _shoot(settings.pen_p, settings.pen_gap, settings.pen_d, settings.pen_order),
    ]
table = EV.table_rows(rows, settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Einheit: Δ in Prozentpunkten (Pp) der Chance der stärksten Seite, zu gewinnen; N in beobachteten Wettbewerben {'je Lage' if settings.design == 'zwei Lagen' else 'in der besten Lage'} "
           f"(Fehlerniveau {settings.alpha} %, Teststärke {settings.power} %, Messanordnung „{settings.design}“).")
cols = st.columns(len(table))
for col, r in zip(cols, table):
    # Delta immer als „mein Wert minus Referenz“ nach DEMO-PLAYBOOK §2: hier nur eine Beschriftung ohne Vergleichsfarbe.
    col.metric(r["model"], f"N = {EV.count(r['n'])}", delta=f"Δ = {EV.pp(r['delta'])}", delta_color="off")
st.dataframe(_fmt_table(table), hide_index=True, width="stretch")
st.plotly_chart(build_compare(table), width="stretch", key="compare_chart")
st.caption("Grau: das Rauschen, näherungsweise der Anteil, in dem die Stärkste nicht gewinnt (100 Pp minus Fairwert; der Fairwert ist hier die Mitte der beiden Lagen und kann von der Chance ohne Verzerrung abweichen, beim Auftritt bei den Voreinstellungen um rund 5 Pp). Farbig: der Lagenunterschied Δ.")
st.plotly_chart(build_needed(table), width="stretch", key="needed_chart")
st.caption("Benötigte beobachtete Wettbewerbe (logarithmische Achse). Die Striche zeigen bei den simulierten Modellen die Spanne bei Δ ± 2 Standardfehler; „unbestimmt“ heißt: kein Lagenunterschied, also nie zu sehen.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Rauschen oder Verzerrung?")
kind, text = EV.verdict_comparison(table)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Saison für Saison, bis die Verzerrung gegen den Zufall steht")
example = EV.example_row(rows, settings.example)
path, n_two = EV.example_path(example, settings)
st.caption(f"Modell: {example.model} (Auswahl in der Seitenleiste). In jeder „Saison“ wird ein Wettbewerb in der besten Lage ({example.good}) und einer in der schlechtesten ({example.bad}) gespielt (je Modell ist ein Wettbewerb etwas anderes: eine Liga-Saison, ein Duell, ein Auftritts-Wettbewerb, ein Elfmeterschießen); "
           "gewürfelt wird mit dem Zufalls-Seed der Seitenleiste und den Chancen der Tabelle. Die Linie zeigt den Unterschied der Siegquoten der Stärksten, das graue Band den Bereich, den der Zufall allein erreicht.")
st.plotly_chart(build_example(path, n_two, example.delta, example.model), width="stretch", key="example_chart")
kind, text = EV.verdict_example(example.model, path, n_two, len(path.seasons), settings.alpha_p)
getattr(st, kind)(text)
horizon = len(path.seasons)
peek = _peeking(example.p_fair, horizon, settings.alpha_p, settings.seed)
st.caption("Die Linie darf das Band früher oder später verlassen als die Formel sagt: N ist die Zahl, bei der der Test in der gewünschten Teststärke einen echten Unterschied findet, nicht der Zeitpunkt in diesem einen Beispiel. "
           f"Wer die Linie jede Saison beobachtet und beim ersten Ausreißer abbricht, macht mehr Fehlalarme als das Fehlerniveau sagt: In {C.PEEK_PATHS} gewürfelten Beispielen ohne jeden Unterschied (beide Lagen mit {EV.pct(example.p_fair)}) "
           f"verlässt die Linie in {EV.pct(peek, 0)} das Band mindestens einmal innerhalb von {horizon} Saisons, bei einem Fehlerniveau von {settings.alpha} %.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Die Rauschgrenze, wie oft findet ein fairer Wettbewerb die Stärkste?")
st.caption(f"Vergleichsmaßstab: eine Liga mit {settings.noise_n} Teilnehmern ohne jede Verzerrung, Stärken auf einer Leiter mit 0,3 logistischer Stufe je Platz (rund 52 Elo-Punkte), Partien ohne Remis, Tabelle nach Punkten, "
           f"Gleichstand an der Spitze zu gleichen Teilen; {C.NOISE_RUNS:,} Ligen je Balken.".replace(",", "."))
noise = _noise(settings.noise_n, settings.seed)
st.plotly_chart(build_noise(noise, settings.noise_games), width="stretch", key="noise_chart")
kind, text = EV.verdict_noise(noise)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 3: Probe der Teststärke, stimmt die Zahl der Formel?")
if math.isinf(n_two) or example.delta <= 0:
    st.info("Das gewählte Modell hat keinen Lagenunterschied (Δ = 0): es gibt keine Teststärke zu prüfen. Die Fehlalarmrate wäre das Fehlerniveau.")
else:
    n_probe = int(EV.needed_for(example, settings.alpha_p, settings.power_p, settings.design))
    sim, exact = _power(example.p_good, example.p_bad, n_probe, settings.alpha_p, settings.design, settings.seed)
    st.caption(f"Modell: {example.model}; N = {EV.count(n_probe)} nach der Formel ({settings.design}), Chancen {EV.pct(example.p_good)} und {EV.pct(example.p_bad)}. "
               f"Der Test wird {C.POWER_REPS:,} Mal gewürfelt; die exakte Rechnung summiert über beide Binomialverteilungen (bis N = {PW.EXACT_N_LIMIT:,}).".replace(",", "."))
    c1, c2 = st.columns(2)
    c1.metric("Gewürfelte Teststärke", EV.pct(sim.rate), delta=f"Ziel {settings.power} %", delta_color="off")
    c2.metric("Exakte Teststärke", EV.pct(exact) if exact is not None else "nicht gerechnet", delta=f"Ziel {settings.power} %", delta_color="off")
    kind, text = EV.verdict_power(sim, settings.power_p, exact)
    getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment 4: Messreihe über die Verzerrungsstärke, vier Modelle in einer Einheit")
st.caption("Je Reihe wächst die Stärke der Verzerrung (Wirkung der Form, Heim-Mittel bei Summe 2,6, Drift oder Anker, Druck); alle anderen Regler stehen auf den Standardwerten, unabhängig von der Seitenleiste. "
           "Spielplan und Auftritt: Mittel über fünf feste Zufalls-Seeds mit je 20.000 Wettbewerben (vorgerechnet in `data/messreihe.json`); Rückspiel und Elfmeterschießen exakt. Nur α, Teststärke und Messanordnung wirken auf die Zahl N.")
series = _sweeps()
rows_n = {name: EV.sweep_needed(rws, settings.alpha_p, settings.power_p, settings.design) for name, (_, rws) in series.items()}
st.plotly_chart(build_sweep(series, rows_n), width="stretch", key="sweep_chart")
kind, text = EV.verdict_sweep({name: [(r["delta"], n) for r, n in zip(rws, rows_n[name])] for name, (_, rws) in series.items()})
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Stärke ist bekannt** (synthetische Wettbewerbe, wahre Stärke wird vorgegeben) | In echten Wettbewerben steht die „stärkste Seite“ nicht fest; Δ und N lassen sich nur an einem Modell mit bekannter Stärke angeben. Die Zahlen sagen nichts über echte Wettbewerbe. | – |
| **Die vier Modelle sind eigene, kompakte Fassungen** mit eigenen Annahmen (Poisson-Tore, Elo-Umrechnung, Markow-Zustand mit Gedächtnis, Drift und Anker, Druck im Rückstand) | Andere Annahmen verschieben Δ, ohne die Einheit zu ändern; die Reihenfolge der Modelle kann sich ändern. Die Modelle sind kleine Nachbauten der Stücke der Linie, nicht deren Code. | Spielplanreihenfolge und Form; Hin- und Rückspiel; Auftrittsreihenfolge und Kampfrichter-Anker; Wann zählt die Reihenfolge? |
| **Δ ist der Unterschied zwischen bester und schlechtester Lage**, nicht der Mittelwert über eine Auslosung | Wird die Lage ausgelost, verteilt der Zufall den Vorteil auf die Seiten, beseitigt ihn aber im einzelnen Wettbewerb nicht. Beim Spielplan ist „bester und schlechtester von 24 Plänen“ eine Auswahl, kein Naturwert. | – |
| **Ein beobachteter Wettbewerb ist je Modell etwas anderes** (Liga mit 28 Partien, Duell mit 2 Spielen, Elfmeterschießen, Auftritts-Wettbewerb) | N lässt sich zwischen den Modellen nur in Wettbewerben vergleichen, nicht im Aufwand: 569 Elfmeterschießen und 1.810 Saisons sind nicht gleich aufwendig zu beobachten. Die Rangfolge der Modelle gilt für die Voreinstellungen und für beste gegen schlechteste von 24 Zuordnungen bzw. 10 Plätzen. | – |
| **Beim Spielplan ist Δ eine Einzelstichprobe einer Auswahl** (beste und schlechteste von 24 Zuordnungen, Messung mit einem Zufalls-Seed) | Bei den Voreinstellungen und 8.000 Läufen liegt Δ mit den Zufalls-Seeds 101 bis 105 zwischen 2,01 und 4,46 Pp (Streuung 0,98 Pp, angezeigter Standardfehler 0,54 Pp), N zwischen 1.922 und 9.485. Die angezeigte Spanne für N ist für den Spielplan zu eng. | – |
| **Der Fairwert ist die Mitte der beiden Lagen** | Die Chance ohne Verzerrung kann davon abweichen (Auftritt bei den Voreinstellungen: Mitte 76,7 %, ohne Drift 81,7 %); die Messanordnung „gegen den Fairwert“ prüft dann gegen die Mitte, nicht gegen die unverzerrte Chance. | – |
| **Ein Wettbewerb liefert ja oder nein** (die Stärkste gewinnt) | Andere Größen (Rang, Punkte, Torverhältnis) liefern mehr als ein Ja oder Nein je Wettbewerb; wie viel das an N spart, ist hier nicht gemessen. | – |
| **Unabhängige Wettbewerbe mit gleicher Verzerrung** | Ändert sich die Verzerrung von Saison zu Saison (Regeländerung, andere Stärken), taugt die Formel nicht; gemeinsame Wiederholung wird nicht modelliert. | – |
| **Normalnäherung der Formel für N** | Bei kleinen N weicht die Teststärke ab; die Probe in Experiment 3 rechnet sie exakt nach. | – |
| **Anreize sind keine Lagen**: absichtliches Verlieren, Mitnahme von Punkten, Draft-Lotterie | Dort entscheidet eine Mannschaft selbst; gemessen wird ein Anreiz, keine Verschiebung der Gewinnchance durch eine Ausgangslage. Sie stehen nicht in dieser Tabelle. | Wann lohnt Verlieren?; Der Swiss Gambit; Draft-Lotterie ohne Tanking |
| **Der Hot-Hand-Messfehler ist ein Messproblem** | Dieselbe Zufallsfolge, anderes Maß: die Gewinnchance einer Lage ändert sich nicht, es gibt kein Δ. Er gehört nicht in die Einheit dieser Tabelle. | Der Hot-Hand-Messfehler |
| **Rauschgrenze mit Partien ohne Remis und gleichmäßiger Leiter** | Mit Remis, Heimvorteil, Form oder anderer Stärkeverteilung ändert sich die Chance der Stärksten an der Spitze. | [Tabelle nach Siegen](https://sebastianhanisch-siegtabelle-demo.streamlit.app/), [Rangintervalle](https://sebastianhanisch-rangintervalle-demo.streamlit.app/) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind vier Modelle mit bekannter wahrer Stärke; gesucht sind je Modell der Lagenunterschied $\Delta$ und die Zahl $N$ beobachteter Wettbewerbe, die ihn gegen den Zufall sichtbar macht. Dieser Block fasst die Modelle zusammen (eine Skizze, nicht die vollständigen Programme); alle Definitionen stehen im Quelltext (`rau_plan.py`, `rau_duel.py`, `rau_judges.py`, `rau_shootout.py`, `rau_power.py`).

| Symbol | Bedeutung |
|---|---|
| $P_{\mathrm{gut}}$, $P_{\mathrm{schlecht}}$ | Chance der stärksten Seite zu gewinnen in der besten bzw. schlechtesten Lage |
| $\Delta$ | Lagenunterschied $P_{\mathrm{gut}} - P_{\mathrm{schlecht}}$ (Anteil; angezeigt in Prozentpunkten) |
| $\bar p$ | Fairwert, hier die Mitte $(P_{\mathrm{gut}} + P_{\mathrm{schlecht}})/2$ der beiden Lagen (nicht die Chance ohne Verzerrung) |
| $\alpha$, $1-\beta$ | Fehlerniveau (zweiseitig) und gewünschte Teststärke |
| $z_q$ | $q$-Quantil der Standardnormalverteilung |
| $N$ | beobachtete Wettbewerbe je Lage |

**Einheit.** Für jedes Modell ist $\Delta = P_{\mathrm{gut}} - P_{\mathrm{schlecht}}$ mit der Chance der wahrhaft stärksten Seite. Ein beobachteter Wettbewerb ist eine Bernoulli-Beobachtung mit Varianz $p(1-p)$.

**Benötigte Zahl, zwei Lagen.** Zweiseitiger Test auf gleiche Anteile mit gemeinsamem Anteil unter der Nullhypothese:

$$N = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{P_{\mathrm{gut}}(1-P_{\mathrm{gut}}) + P_{\mathrm{schlecht}}(1-P_{\mathrm{schlecht}})}\right)^2}{\Delta^2}$$

Da $P_{\mathrm{gut}}(1-P_{\mathrm{gut}}) + P_{\mathrm{schlecht}}(1-P_{\mathrm{schlecht}}) = 2\bar p(1-\bar p) - \Delta^2/2$ gilt, wächst $N$ bei festem $\bar p$ etwa mit $1/\Delta^2$.

**Gegen den Fairwert.** Einstichproben-Test gegen $p_0 = \bar p$ mit $p_1 = P_{\mathrm{gut}}$:

$$N = \left(\frac{z_{1-\alpha/2}\sqrt{p_0(1-p_0)} + z_{1-\beta}\sqrt{p_1(1-p_1)}}{p_1 - p_0}\right)^2$$

**Exakte Teststärke** (Gegenprobe der Formel), $X_1 \sim \mathrm{Bin}(N, P_{\mathrm{gut}})$, $X_2 \sim \mathrm{Bin}(N, P_{\mathrm{schlecht}})$:

$$1-\beta_{\mathrm{exakt}} = \sum_{x_1, x_2} \Pr(X_1 = x_1)\,\Pr(X_2 = x_2)\,\mathbf{1}\left[\,|x_1 - x_2|/N \ge z_{1-\alpha/2}\sqrt{2\hat p(1-\hat p)/N}\,\right], \quad \hat p = \frac{x_1 + x_2}{2N}$$

**Spielplan und Form.** Elo-Erwartung $e = 1/(1 + 10^{-(r_a + m s_a - r_b - m s_b)/400})$, Remisanteil $d = \min(d_{\max}, 2\min(e, 1-e))$, Sieg $e - d/2$. Zustand $s \in \{-1, 0, +1\}$: nach jeder Partie bleibt er mit Wahrscheinlichkeit $\rho$, sonst wird er auf das Ergebnis gesetzt. $P_{\mathrm{gut}}$ und $P_{\mathrm{schlecht}}$ sind die Titelchancen der stärksten Mannschaft in der besten und schlechtesten von 24 Zuordnungen der Mannschaften zur Zirkeltafel (ausgewählt an der ersten Hälfte der Läufe nach den mittleren Punkten, gemessen an der zweiten).

**Hin- und Rückspiel.** Mit der Differenz $A$ der Auswärtstore und $H$ der Heimtore der stärkeren minus der schwächeren Mannschaft (unabhängige Skellam-Verteilungen) ist die Tor-Summe $D = A + H$:

$$P(\text{stärkere kommt weiter}) = \Pr(D > 0) + \Pr(D = 0)\cdot q_{\mathrm{Verl.}}$$

Dabei ist $q_{\mathrm{Verl.}}$ die Chance in Verlängerung (ein Drittel der Torrate im Stadion des Rückspiel-Gastgebers) und Elfmeterschießen; mit Auswärtstorregel entscheidet bei $D = 0$ zuerst $\mathrm{sign}(A)$.

**Auftrittsreihenfolge.** Note von Richter $j$ für den Auftretenden auf Platz $k$:

$$x_{k,j} = q + \sigma e_{k,j} + \beta\left(k - \tfrac{n-1}{2}\right) + \gamma A_{k-1}, \qquad A_k = \tfrac{1}{J}\sum_j x_{k,j}$$

Die höchste Wertung $A_k$ gewinnt; $P_{\mathrm{gut}}$ und $P_{\mathrm{schlecht}}$ gehören zur besten und schlechtesten Auftrittsnummer der Qualitätsbesten.

**Elfmeterschießen.** Ein Schütze trifft mit $p$, im Rückstand mit $p - d$. Nach der Regelzeit (R Schüsse je Mannschaft) entscheidet bei Gleichstand der Sudden Death; bei ABAB beginnt die erste Mannschaft jedes Paar und gewinnt es mit $a_1 = p_A(1 - (p_B - d))$ gegen $b_1 = (1 - p_A)p_B$, also insgesamt mit $a_1/(a_1 + b_1)$.

**Rauschgrenze.** Chance der Stärksten an der Spitze einer fairen Liga mit $k$ Partien je Paar, Stärken $s_i = -0{,}3\,i$ (logistisch), Siegchance $1/(1+e^{-(s_i - s_j)})$, Gleichstand zu gleichen Teilen.

**Literatur** (nicht im Volltext gelesen, Angaben nicht geprüft): Maher (1982) zu Poisson-Toren im Fußball; Bruine de Bruin (2005) zu Auftrittsplätzen bei Noten (nach Suchausschnitt); Miller und Sanjurjo (2018) zum Hot-Hand-Messfehler; Fleiss, Levin und Paik: Statistical Methods for Rates and Proportions (Formeln der benötigten Stichprobengröße für Anteile).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Wettbewerbsverzerrung: Reihenfolge und Anreize](https://sebastianhanisch.net/konzepte-wettbewerbsverzerrung.html)."
)
