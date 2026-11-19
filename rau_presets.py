"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Der Schlüssel eines Reglers im Session State ist zugleich der Name des Feldes in `rau_evaluation.Settings` und der Schlüssel in den Presets."""

import random
from dataclasses import dataclass

import streamlit as st

import rau_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type | object
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None        # Rastermaß des Reglers: Werte aus der Adresszeile werden darauf eingerastet
    options: tuple | None = None     # erlaubte Werte (Auswahlregler); Werte aus der Adresszeile rasten auf den nächsten ein bzw. fallen auf den Standard zurück


def to_bool(value) -> bool:
    """Adresszeile: „True“/„False“ (so schreibt sync_query_params den Wert) und „1“/„0“."""
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(value)


def snap(value, spec: SettingSpec):
    """Begrenzt einen Wert aus der Adresszeile auf die Reglergrenzen und rastet ihn auf Raster oder Optionen ein."""
    if spec.options is not None:
        if all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in spec.options):
            return min(spec.options, key=lambda o: abs(o - value))
        return value if value in spec.options else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, max(spec.lo, value))
        return int(round(value)) if spec.caster is int else round(value, 4)
    return value


SETTING_SPECS = {
    "seed": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "runs": SettingSpec("runs", int, C.DEFAULT_RUNS, options=C.RUNS_OPTIONS),
    "alpha": SettingSpec("alpha", int, C.DEFAULT_ALPHA, options=C.ALPHA_OPTIONS),
    "power": SettingSpec("power", int, C.DEFAULT_POWER, options=C.POWER_OPTIONS),
    "design": SettingSpec("design", str, C.DEFAULT_DESIGN, options=C.DESIGNS),
    "example": SettingSpec("ex", str, C.DEFAULT_EXAMPLE, options=C.MODELS),
    "plan_n": SettingSpec("pn", int, C.DEFAULT_PLAN_N, options=C.PLAN_N_OPTIONS),
    "plan_gap": SettingSpec("pg", int, C.DEFAULT_PLAN_GAP, C.PLAN_GAP_MIN, C.PLAN_GAP_MAX, C.PLAN_GAP_STEP),
    "form_m": SettingSpec("fm", int, C.DEFAULT_FORM_M, C.FORM_M_MIN, C.FORM_M_MAX, C.FORM_M_STEP),
    "form_rho": SettingSpec("fr", float, C.DEFAULT_FORM_RHO, C.FORM_RHO_MIN, C.FORM_RHO_MAX, C.FORM_RHO_STEP),
    "plan_double": SettingSpec("pd", to_bool, C.DEFAULT_PLAN_DOUBLE),
    "lam_home": SettingSpec("lh", float, C.DEFAULT_LAM_HOME, C.LAM_HOME_MIN, C.LAM_HOME_MAX, C.LAM_STEP),
    "lam_away": SettingSpec("la", float, C.DEFAULT_LAM_AWAY, C.LAM_AWAY_MIN, C.LAM_AWAY_MAX, C.LAM_STEP),
    "duel_elo": SettingSpec("de", int, C.DEFAULT_DUEL_ELO, C.DUEL_ELO_MIN, C.DUEL_ELO_MAX, C.DUEL_ELO_STEP),
    "away_rule": SettingSpec("ar", to_bool, C.DEFAULT_AWAY_RULE),
    "extra_time": SettingSpec("et", to_bool, C.DEFAULT_EXTRA_TIME),
    "perf_n": SettingSpec("an", int, C.DEFAULT_PERF_N, C.PERF_N_MIN, C.PERF_N_MAX, 1),
    "judges": SettingSpec("jd", int, C.DEFAULT_JUDGES, C.JUDGES_MIN, C.JUDGES_MAX, 1),
    "drift": SettingSpec("dr", float, C.DEFAULT_DRIFT, C.DRIFT_MIN, C.DRIFT_MAX, C.DRIFT_STEP),
    "anchor": SettingSpec("ac", float, C.DEFAULT_ANCHOR, C.ANCHOR_MIN, C.ANCHOR_MAX, C.ANCHOR_STEP),
    "pen_p": SettingSpec("pp", float, C.DEFAULT_PEN_P, C.PEN_P_MIN, C.PEN_P_MAX, C.PEN_P_STEP),
    "pen_gap": SettingSpec("pgp", float, C.DEFAULT_PEN_GAP, C.PEN_GAP_MIN, C.PEN_GAP_MAX, C.PEN_GAP_STEP),
    "pen_d": SettingSpec("pd2", float, C.DEFAULT_PEN_D, C.PEN_D_MIN, C.PEN_D_MAX, C.PEN_D_STEP),
    "pen_order": SettingSpec("po", str, C.DEFAULT_PEN_ORDER, options=C.PEN_ORDERS),
    "noise_n": SettingSpec("nn", int, C.DEFAULT_NOISE_N, C.NOISE_N_MIN, C.NOISE_N_MAX, 1),
    "noise_games": SettingSpec("ng", int, C.DEFAULT_NOISE_GAMES, options=C.NOISE_GAMES_OPTIONS),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = snap(value, spec)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)
