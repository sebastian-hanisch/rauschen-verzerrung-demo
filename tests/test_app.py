"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button und jeden Knopf, fährt jeden Regler an beide Grenzen (und alle Minima und alle
Maxima zugleich) und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Diagrammen ohne key und Abstürze an den Reglerrändern)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rau_constants as C
import rau_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} fehlt")


def _edges(key):
    spec = P.SETTING_SPECS[key]
    if spec.options is not None:
        return spec.options[0], spec.options[-1]
    if isinstance(spec.default, bool):
        return False, True
    return spec.lo, spec.hi


def test_app_loads_without_exception_and_shows_every_section():
    at = _fresh_app()
    texts = " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.subheader)
    for part in ("Ergebnis", "Kernfrage", "Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Wo die Annahmen enden"):
        assert part in texts, part
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().sidebar.button]
    assert len(labels) == len(C.PRESETS) + 1
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.sidebar.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", list(P.SETTING_SPECS))
@pytest.mark.parametrize("edge", [0, 1])
def test_every_control_at_both_edges_does_not_raise(key, edge):
    at = _fresh_app()
    _regler(at, key).set_value(_edges(key)[edge]).run(timeout=TIMEOUT)
    assert not at.exception, (key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", [0, 1])
def test_all_minima_and_all_maxima_together_do_not_raise(edge):
    at = _fresh_app()
    for key in P.SETTING_SPECS:
        if key == "seed":
            continue
        _regler(at, key).set_value(_edges(key)[edge])
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_seed_button_changes_the_seed_only():
    at = _fresh_app()
    before = at.sidebar.number_input(key="seed").value
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.number_input(key="seed").value != before or before == 0


def test_the_four_metrics_and_the_table_are_shown():
    at = _fresh_app()
    assert len(at.metric) >= 4
    frames = [d for d in at.dataframe]
    assert frames and "Lagenunterschied Δ" in list(frames[0].value.columns)
    assert "Fairwert (Mitte der beiden Lagen)" in list(frames[0].value.columns)                # der Fairwert ist die Mitte der Lagen, nicht die Chance ohne Verzerrung
    assert any("Einzelstichprobe" in e.value for e in [*at.success, *at.info])                            # Spielplan-Hinweis im Verdikt der Kernfrage (Voreinstellungen)
    assert list(frames[0].value["Modell"]) == list(C.MODELS)


def test_every_preset_shows_the_verdict_of_its_own_settings():
    """Verdikt- und Hinweistexte sind Behauptungen (DEMO-PLAYBOOK §11a Punkt 7): jedes Preset durchläuft die App und zeigt, was seine Zahlen sagen."""
    expectations = {
        "Standardfall (Voreinstellung)": ("Am leichtesten fällt **Auftrittsreihenfolge** auf", "am schwersten **Hin- und Rückspiel**"),
        "Nur Rauschen: keine Verzerrung": ("In allen vier Modellen ist der Lagenunterschied Δ null",),
        "Starke Verzerrung": ("Am leichtesten fällt **Auftrittsreihenfolge** auf", "am schwersten **Hin- und Rückspiel**"),
        "Strenger Test: 1 % Fehlerniveau, 95 % Teststärke": ("N = 36.669",),
        "Doppelrunde und mehr Partien je Paar": ("Am leichtesten fällt **Auftrittsreihenfolge** auf",),
        "Messen gegen den bekannten Fairwert": ("N = 32.136",),
    }
    assert set(expectations) == set(C.PRESETS)
    for name, parts in expectations.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        shown = " ".join(e.value for e in [*at.success, *at.info, *at.warning]) + " ".join(m.value for m in at.metric) + " ".join(c.value for c in at.caption)
        for part in parts:
            assert part in shown, (name, part)


def test_noise_only_preset_shows_the_example_without_a_difference():
    at = _fresh_app()
    at.sidebar.button(key="preset_Nur Rauschen: keine Verzerrung").click().run(timeout=TIMEOUT)
    shown = " ".join(e.value for e in [*at.info, *at.success, *at.warning])
    assert "keinen Lagenunterschied" in shown and "keine Teststärke zu prüfen" in shown
