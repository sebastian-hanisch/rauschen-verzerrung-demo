"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Raster der Messreihen und Presets.

Rechenaufwand am Regler-Maximum (DEMO-PLAYBOOK §1c): Das Spielplan-Modell kostet O(Zuordnungen · Läufe · Spieltage · Partien je Spieltag) = 24 · 20.000 · 14 · 4
Partien (Doppelrunde mit 8 Mannschaften), rund 1,5 bis 3 s; die Auftritts-Simulation O(Plätze · Läufe · Auftretende) = 12 · 20.000 · 12, unter einer Sekunde; das
Hin- und Rückspiel und das Elfmeterschießen sind exakt und kosten Millisekunden; die Rauschgrenze O(Paare · Partien · Läufe) = 66 · 40 · 4.000 = 10,6 Millionen Zufallszahlen
bei 12 Teilnehmern und 40 Partien je Paar, rund eine halbe Sekunde je Punkt der Kurve.
"""

SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Messung und Test ---
RUNS_OPTIONS = (2000, 5000, 8000, 20000)
DEFAULT_RUNS = 8000
ALPHA_OPTIONS = (1, 2, 5, 10)             # Fehlerniveau in Prozent (zweiseitig)
DEFAULT_ALPHA = 5
POWER_OPTIONS = (70, 80, 90, 95, 99)      # gewünschte Teststärke in Prozent
DEFAULT_POWER = 80
DESIGNS = ("zwei Lagen", "gegen den Fairwert")
DEFAULT_DESIGN = "zwei Lagen"
MODELS = ("Spielplan und Form", "Hin- und Rückspiel", "Auftrittsreihenfolge", "Elfmeterschießen")
DEFAULT_EXAMPLE = "Hin- und Rückspiel"

# --- Modell 1: Spielplan und Form ---
PLAN_N_OPTIONS = (4, 6, 8)
DEFAULT_PLAN_N = 8
PLAN_GAP_MIN, PLAN_GAP_MAX, PLAN_GAP_STEP, DEFAULT_PLAN_GAP = 0, 600, 50, 300     # Elo-Abstand Stärkster gegen Schwächsten
FORM_M_MIN, FORM_M_MAX, FORM_M_STEP, DEFAULT_FORM_M = 0, 150, 25, 50              # Wirkung der Form in Elo-Punkten
FORM_RHO_MIN, FORM_RHO_MAX, FORM_RHO_STEP, DEFAULT_FORM_RHO = 0.0, 0.95, 0.05, 0.5  # Gedächtnis des Zustands
DEFAULT_PLAN_DOUBLE = False
PLAN_DRAW_MAX = 0.25                       # höchster Remisanteil einer Partie (Annahme)
PLAN_ASSIGNMENTS = 24                      # Zahl der Zuordnungen je Messung (bei n = 4 sind das alle 4! = 24)

# --- Modell 2: Hin- und Rückspiel ---
LAM_HOME_MIN, LAM_HOME_MAX, LAM_STEP, DEFAULT_LAM_HOME = 0.5, 2.5, 0.05, 1.5
LAM_AWAY_MIN, LAM_AWAY_MAX, DEFAULT_LAM_AWAY = 0.3, 2.0, 1.1
DUEL_ELO_MIN, DUEL_ELO_MAX, DUEL_ELO_STEP, DEFAULT_DUEL_ELO = 0, 300, 25, 100      # Elo-Abstand der stärkeren zur schwächeren Mannschaft
DEFAULT_AWAY_RULE, DEFAULT_EXTRA_TIME = False, True
DUEL_PEN_STRONG = 0.5                      # Elfmeterschießen: Chance der stärkeren Mannschaft (Annahme)

# --- Modell 3: Auftrittsreihenfolge ---
PERF_N_MIN, PERF_N_MAX, DEFAULT_PERF_N = 3, 12, 10
JUDGES_MIN, JUDGES_MAX, DEFAULT_JUDGES = 1, 9, 5
DRIFT_MIN, DRIFT_MAX, DRIFT_STEP, DEFAULT_DRIFT = 0.0, 0.15, 0.01, 0.05           # Notenpunkte je Platz
ANCHOR_MIN, ANCHOR_MAX, ANCHOR_STEP, DEFAULT_ANCHOR = -0.6, 0.6, 0.1, 0.0         # Angleichung (+) oder Kontrast (-)
JUDGE_SIGMA = 0.5                          # Rauschen je Note (Annahme)

# --- Modell 4: Elfmeterschießen ---
PEN_P_MIN, PEN_P_MAX, PEN_P_STEP, DEFAULT_PEN_P = 0.5, 0.95, 0.05, 0.75           # Trefferwahrscheinlichkeit der stärkeren Mannschaft
PEN_GAP_MIN, PEN_GAP_MAX, PEN_GAP_STEP, DEFAULT_PEN_GAP = 0.0, 0.2, 0.01, 0.05    # Vorsprung der stärkeren Mannschaft
PEN_D_MIN, PEN_D_MAX, PEN_D_STEP, DEFAULT_PEN_D = 0.0, 0.25, 0.01, 0.05           # Druck: Abschlag im Rückstand
PEN_ORDERS = ("ABAB", "ABBA")
DEFAULT_PEN_ORDER = "ABAB"
PEN_SHOTS = 5

# --- Rauschgrenze ---
NOISE_N_MIN, NOISE_N_MAX, DEFAULT_NOISE_N = 4, 12, 8
NOISE_GAMES_OPTIONS = (1, 2, 4, 10, 20, 40)
DEFAULT_NOISE_GAMES = 1
NOISE_RUNS = 4000

# --- Messreihen (vorgerechnet in data/messreihe.json, Erzeuger tools/make_messreihe.py) ---
SWEEP_SEEDS = (101, 102, 103, 104, 105)
SWEEP_RUNS = 20000
SWEEP_FORM_M = (0, 25, 50, 75, 100, 150)
SWEEP_FORM_RHO = (0.0, 0.5, 0.8)
SWEEP_DRIFT = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
SWEEP_ANCHOR = (-0.6, -0.3, 0.3, 0.6)
SWEEP_LAM_HOME = (1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)    # Auswärts-Mittel = 2,6 minus Heim-Mittel
SWEEP_LAM_TOTAL = 2.6
SWEEP_PEN_D = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)

# Probe der Teststärke und Beispielweg
POWER_REPS = 4000
EXAMPLE_MAX_SEASONS = 3000
PEEK_PATHS = 400                # Beispielwege ohne Unterschied für die Zählung der Fehlalarme bei ständigem Hinsehen

COLORS = {
    "Spielplan und Form": "#2F6B65",
    "Hin- und Rückspiel": "#c9743b",
    "Auftrittsreihenfolge": "#4c78a8",
    "Elfmeterschießen": "#9467bd",
}
COLOR_GREY = "rgba(150,150,150,0.8)"
COLOR_LINE = "#4c78a8"

LAGE_LABELS = {
    "Spielplan und Form": ("beste Zuordnung", "schlechteste Zuordnung"),
    "Hin- und Rückspiel": ("Rückspiel zu Hause", "Hinspiel zu Hause"),
    "Auftrittsreihenfolge": ("beste Auftrittsnummer", "schlechteste Auftrittsnummer"),
    "Elfmeterschießen": ("schießt zuerst", "schießt als zweite"),
}

_BASE = {
    "seed": DEFAULT_SEED, "runs": DEFAULT_RUNS, "alpha": DEFAULT_ALPHA, "power": DEFAULT_POWER, "design": DEFAULT_DESIGN, "example": DEFAULT_EXAMPLE,
    "plan_n": DEFAULT_PLAN_N, "plan_gap": DEFAULT_PLAN_GAP, "form_m": DEFAULT_FORM_M, "form_rho": DEFAULT_FORM_RHO, "plan_double": DEFAULT_PLAN_DOUBLE,
    "lam_home": DEFAULT_LAM_HOME, "lam_away": DEFAULT_LAM_AWAY, "duel_elo": DEFAULT_DUEL_ELO, "away_rule": DEFAULT_AWAY_RULE, "extra_time": DEFAULT_EXTRA_TIME,
    "perf_n": DEFAULT_PERF_N, "judges": DEFAULT_JUDGES, "drift": DEFAULT_DRIFT, "anchor": DEFAULT_ANCHOR,
    "pen_p": DEFAULT_PEN_P, "pen_gap": DEFAULT_PEN_GAP, "pen_d": DEFAULT_PEN_D, "pen_order": DEFAULT_PEN_ORDER,
    "noise_n": DEFAULT_NOISE_N, "noise_games": DEFAULT_NOISE_GAMES,
}

PRESETS = {
    "Standardfall (Voreinstellung)": dict(_BASE),
    "Nur Rauschen: keine Verzerrung": {**_BASE, "form_m": 0, "lam_home": 1.3, "lam_away": 1.3, "drift": 0.0, "anchor": 0.0, "pen_d": 0.0, "example": "Spielplan und Form"},
    "Starke Verzerrung": {**_BASE, "form_m": 100, "form_rho": 0.0, "lam_home": 1.7, "lam_away": 0.9, "drift": 0.10, "pen_d": 0.15, "example": "Elfmeterschießen"},
    "Strenger Test: 1 % Fehlerniveau, 95 % Teststärke": {**_BASE, "alpha": 1, "power": 95},
    "Doppelrunde und mehr Partien je Paar": {**_BASE, "plan_double": True, "noise_games": 4},
    "Messen gegen den bekannten Fairwert": {**_BASE, "design": "gegen den Fairwert"},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Alle Regler auf ihrem Standardwert: Spielplan mit Form (m = 50, ρ = 0,5), Rückspiel mit Heim 1,5 und Auswärts 1,1, Auftritts-Drift +0,05, Elfmeterdruck 0,05; Test mit 5 % Fehlerniveau und 80 % Teststärke.",
    "Nur Rauschen: keine Verzerrung": "Alle vier Verzerrungen ausgeschaltet (Form 0, Heim gleich Auswärts, keine Drift und kein Anker, kein Druck). Jeder Lagenunterschied ist dann null; was bleibt, ist das Rauschen.",
    "Starke Verzerrung": "Alle vier Verzerrungen deutlich stärker: Form 100 ohne Gedächtnis, Heim 1,7 und Auswärts 0,9, Drift +0,10, Druck 0,15. Das Beispiel zeigt das Elfmeterschießen.",
    "Strenger Test: 1 % Fehlerniveau, 95 % Teststärke": "Wie der Standardfall, aber ein strengerer Test: weniger Fehlalarme (1 %) und eine höhere Chance, einen echten Unterschied zu finden (95 %). Das verlangt mehr Wettbewerbe.",
    "Doppelrunde und mehr Partien je Paar": "Wie der Standardfall, aber der Spielplan ist eine Doppelrunde und die Rauschgrenze rechnet mit 4 Partien je Paar.",
    "Messen gegen den bekannten Fairwert": "Wie der Standardfall, aber geprüft wird nur die beste Lage gegen den als bekannt angenommenen Fairwert (die Mitte der beiden Lagen), nicht zwei Lagen gegeneinander. Das setzt voraus, dass man den Fairwert kennt, und braucht in dieser Lage etwa doppelt so viele Wettbewerbe wie je Lage bei zwei Lagen.",
}
